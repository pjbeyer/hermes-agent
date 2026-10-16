"""1Password (`op` CLI) secret source.

Users map env-var names to ``op://vault/item/field`` references in
``secrets.onepassword.env``; each is resolved with one ``op read -- <ref>``
call using whatever auth the user's ``op`` already has (``OP_SERVICE_ACCOUNT_TOKEN``
headless, ``OP_SESSION_*`` interactive) — Hermes never authenticates on the
user's behalf, and failures never block startup. Complete pulls are cached
in-process and under ``<hermes_home>/cache/op_cache.json`` (values only; auth
material is fingerprinted, never stored).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # tests monkeypatch ``op.subprocess.run``; production goes through run_cli
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources._cache import CachedFetch, SecretCache, fingerprint as _fingerprint
from agent.secret_sources.base import (
    ErrorKind, FetchResult, SecretSource, classify_cli_error, coerce_float,
    get_source_environment, is_valid_env_name, run_cli,
)

logger = logging.getLogger(__name__)

_OP_RUN_TIMEOUT = 30

# A vault/network blip fails every reference in a single run (observed: all 32
# references failing within the same second). One attempt therefore turns a
# momentary fault into a fully degraded pull, so transient kinds are retried.
_OP_READ_ATTEMPTS = 3
_OP_RETRY_BACKOFF_SECONDS = 0.5

# Only these kinds are worth retrying: they describe the transport, not the
# request. A bad reference, a rejected credential, or an empty value is
# deterministic and must fail immediately.
_TRANSIENT_KINDS = frozenset({ErrorKind.NETWORK, ErrorKind.TIMEOUT})

# `op` itself reads OP_SERVICE_ACCOUNT_TOKEN; `service_account_token_env` lets
# the user source it from another name, and _op_child_env normalizes it back.
_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"

# Minimal allowlisted child env (never the full post-dotenv os.environ, which
# holds every provider credential). OP_SESSION_* and the token are added
# dynamically in _op_child_env().
_OP_ENV_ALLOWLIST = (
    "PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
    "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR", "OP_CONFIG_DIR",
    "OP_ACCOUNT", "OP_CONNECT_HOST", "OP_CONNECT_TOKEN",
    # Lets a user skip op's desktop-app integration probe (which can hang with
    # no timeout on a wedged desktop container) and go straight to token auth.
    "OP_LOAD_DESKTOP_APP_SETTINGS",
)

# L1 key folds in str(home_path) so a HERMES_HOME switch inside one long-lived
# process (the gateway) can't return another profile's secrets. The disk key
# omits home because the file already lives under <home>/cache/.
_CacheKey = tuple[str, str, str, str]  # (auth_fp, account, home, refs_fp)
_DISK_CACHE_BASENAME = "op_cache.json"


def _disk_key_str(cache_key: _CacheKey) -> str:
    auth_fp, account, _home, refs_fp = cache_key
    return f"{auth_fp}|{account}|{refs_fp}"


_STORE: SecretCache[_CacheKey] = SecretCache(_DISK_CACHE_BASENAME, key_serializer=_disk_key_str)
_CACHE = _STORE.memory  # tests flush L1 directly

_MISSING_BINARY_HINT = (
    "Install the 1Password CLI (https://developer.1password.com/docs/cli/get-started/) "
    "or set secrets.onepassword.binary_path."
)

# First matching rule wins, so a transport failure must be recognised BEFORE the
# authentication rule. The observed outage reports both in one message:
#
#   error initializing client: Authentication: (failed to auth.LookupAuth),
#   Network: (failed to request...)
#
# `op` cannot distinguish "the token was rejected" from "the request never left
# the machine", so it reports both. Matching `authentication` first classified a
# transport outage as AUTH_FAILED, which is not retryable — the retry would have
# never fired on the very failure it exists for. Ordering NETWORK ahead of
# AUTH_FAILED is also the conservative reading: a genuinely rejected credential
# still fails on every attempt, so treating an ambiguous message as transient
# costs at most a short bounded delay, whereas treating a transport blip as a
# credential fault loses the whole pull.
_OP_ERROR_RULES = (
    (ErrorKind.TIMEOUT, ("timed out",)),
    (ErrorKind.BINARY_MISSING, ("not found on path", "not an executable", "failed to invoke")),
    (ErrorKind.NETWORK, ("network", "connection", "resolve host", "dns",
                         "failed to request", "failed to auth.lookupauth")),
    (ErrorKind.AUTH_FAILED, ("unauthorized", "not signed in", "session expired",
                             "authentication", "401", "403")),
    (ErrorKind.EMPTY_VALUE, ("empty value",)),
)


def _classify_op_error(message: str) -> ErrorKind:
    return classify_cli_error(message, _OP_ERROR_RULES)


def _validate_references(references: Optional[dict[str, str]]) -> tuple[dict[str, str], list[str]]:
    """``(valid_refs, warnings)``: keep valid env names bound to stripped ``op://`` strings."""
    valid: dict[str, str] = {}
    warnings: list[str] = []
    for name, ref in (references or {}).items():
        if not is_valid_env_name(name):
            warnings.append(f"Skipping {name!r}: not a valid env-var name")
        elif not isinstance(ref, str):
            warnings.append(f"Skipping {name!r}: reference is not a string")
        elif not ref.strip().startswith("op://"):
            warnings.append(f"Skipping {name!r}: {ref!r} is not an op:// secret reference")
        else:
            valid[name] = ref.strip()
    return valid, warnings


def _auth_fingerprint(token_env: str) -> str:
    """SHA-256 prefix over everything `op` would authenticate with (token, account,
    Connect host/token, ``OP_SESSION_*``), so a new identity never sees old cached values."""
    source_env = get_source_environment()
    parts: list[str] = [f"{label}={source_env.get(var, '')}" for label, var in (
        ("token", token_env), ("account", "OP_ACCOUNT"),
        ("connect_host", "OP_CONNECT_HOST"), ("connect_token", "OP_CONNECT_TOKEN"))]
    parts += [f"{key}={source_env[key]}" for key in sorted(source_env) if key.startswith("OP_SESSION_")]
    return _fingerprint("\n".join(parts))


def _refs_fingerprint(references: dict[str, str]) -> str:
    return _fingerprint("\n".join(f"{name}={references[name]}" for name in sorted(references)))


def find_op(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``op`` binary, or None. A pinned ``binary_path`` is used
    verbatim — pinned-but-missing returns None rather than falling back to PATH."""
    found = binary_path or shutil.which("op")
    if not found or (binary_path and not os.access(binary_path, os.X_OK)):
        return None
    return Path(found)


def _scrub(text: str) -> str:
    """Full ECMA-48 ANSI strip (so a control sequence can't hide text after a redaction marker) + trim."""
    from tools.ansi_strip import strip_ansi

    return strip_ansi(text).replace("\x1b", "").strip()


def _op_child_env(token_value: str) -> dict[str, str]:
    source_env = get_source_environment()
    env = {k: source_env[k] for k in _OP_ENV_ALLOWLIST if k in source_env}
    env.update((k, v) for k, v in source_env.items() if k.startswith("OP_SESSION_"))
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    env["NO_COLOR"] = "1"
    return env


def _run_op_read(op: Path, reference: str, *, account: str = "", token_value: str = "") -> str:
    """Resolve one ``op://`` reference; raises ``RuntimeError`` on any failure, including
    an exit-0 empty value (applying it would clobber a good credential with ``""``)."""
    cmd: list[str] = [str(op), "read"]
    if account:
        cmd += ["--account", account]
    cmd += ["--", reference]  # `--` so a reference can never parse as an op flag

    proc = run_cli(cmd, env=_op_child_env(token_value), timeout=_OP_RUN_TIMEOUT, label="op",
                   timeout_message=f"op read timed out after {_OP_RUN_TIMEOUT}s for {reference!r}", stdin=None)

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        if err:
            raise RuntimeError(f"op read failed for {reference!r}: {err}")
        raise RuntimeError(f"op read exited {proc.returncode} for {reference!r}")

    # Strip only op's trailing newline so intentional edge spaces survive.
    value = (proc.stdout or "").rstrip("\r\n")
    if not value.strip():
        raise RuntimeError(f"op read returned an empty value for {reference!r}")
    return value


def _read_with_retry(op: Path, reference: str, *, account: str = "", token_value: str = "",
                     attempts: int = _OP_READ_ATTEMPTS,
                     sleep: Optional[Callable[[float], None]] = None) -> str:
    """Resolve one reference, retrying only *transient* failures.

    A vault/network blip fails every reference in the same run, so without a
    retry a single transient fault degrades the whole pull. Retrying is safe for
    transport faults and pointless (or harmful) for the rest: a bad reference or
    rejected credential will fail identically on every attempt, and retrying an
    auth failure just delays a failure the user must fix.

    Sleeps with linear backoff between attempts; ``sleep`` is injectable so tests
    don't wait.
    """
    _sleep = sleep or time.sleep
    last: RuntimeError = RuntimeError(f"op read produced no result for {reference!r}")
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return _run_op_read(op, reference, account=account, token_value=token_value)
        except RuntimeError as exc:
            last = exc
            if attempt >= attempts or _classify_op_error(str(exc)) not in _TRANSIENT_KINDS:
                raise
            _sleep(_OP_RETRY_BACKOFF_SECONDS * attempt)
    raise last


def fetch_onepassword_secrets(
    *, references: dict[str, str], account: str = "", token_env: str = _DEFAULT_TOKEN_ENV,
    binary: Optional[Path] = None, binary_path: str = "", use_cache: bool = True,
    cache_ttl_seconds: float = 300, home_path: Optional[Path] = None,
) -> tuple[dict[str, str], list[str]]:
    """Resolve ``references`` (name → ``op://…``) to ``(secrets, warnings)``.

    Raises ``RuntimeError`` only when no ``op`` binary is available; per-ref
    failures become warnings. Only a complete, error-free pull is cached, so a
    transient auth failure isn't frozen in for the whole TTL window.

    Callers that need to know whether the pull was complete should use
    :func:`fetch_onepassword_result`, which also reports which references could
    not be resolved. This tuple-returning form is retained for compatibility.
    """
    result = fetch_onepassword_result(
        references=references, account=account, token_env=token_env, binary=binary,
        binary_path=binary_path, use_cache=use_cache, cache_ttl_seconds=cache_ttl_seconds,
        home_path=home_path)
    return result.secrets, result.warnings


def fetch_onepassword_result(
    *, references: dict[str, str], account: str = "", token_env: str = _DEFAULT_TOKEN_ENV,
    binary: Optional[Path] = None, binary_path: str = "", use_cache: bool = True,
    cache_ttl_seconds: float = 300, home_path: Optional[Path] = None,
    retry_sleep: Optional[Callable[[float], None]] = None,
) -> FetchResult:
    """Resolve ``references`` into a :class:`FetchResult`.

    Unlike the tuple form this distinguishes a *complete* pull from a *degraded*
    one: references that could not be resolved are recorded in
    ``result.unresolved`` (name → reason), and ``result.partial`` is True when
    some but not all resolved. Raises ``RuntimeError`` only when no ``op`` binary
    is available.
    """
    result = FetchResult()
    valid, warnings = _validate_references(references)
    result.warnings.extend(warnings)
    if not valid:
        return result

    token_value = get_source_environment().get(token_env, "").strip()
    cache_key: _CacheKey = (_auth_fingerprint(token_env), account or "",
                            str(home_path) if home_path is not None else "", _refs_fingerprint(valid))

    if use_cache:
        cached = _STORE.lookup(cache_key, cache_ttl_seconds, home_path)
        if cached is not None:
            # A cache hit only exists for a complete pull (see the store guard
            # below), so nothing is unresolved.
            result.secrets = dict(cached.secrets)
            return result

    op = binary or find_op(binary_path)
    if op is None:
        raise RuntimeError("op CLI not found.  Install the 1Password CLI "
                           "(https://developer.1password.com/docs/cli/get-started/) or set "
                           "secrets.onepassword.binary_path to its absolute location.")

    for name in sorted(valid):
        try:
            result.secrets[name] = _read_with_retry(
                op, valid[name], account=account, token_value=token_value, sleep=retry_sleep)
        except RuntimeError as exc:
            # Record BOTH the human-readable warning (kept for existing consumers
            # and logs) and the structured entry, so a caller never has to parse
            # warning text to learn which credential is missing.
            result.warnings.append(str(exc))
            result.unresolved[name] = _scrub(str(exc))[:200]

    if use_cache and not result.unresolved and result.secrets:
        _STORE.store(cache_key, CachedFetch(secrets=dict(result.secrets), fetched_at=time.time()),
                     cache_ttl_seconds, home_path)

    return result


def _missing_binary_error(binary_path: str) -> str:
    if binary_path:
        return f"secrets.onepassword.binary_path ({binary_path!r}) is not an executable op binary."
    return ("secrets.onepassword.enabled is true but the op CLI was not found on PATH.  Install it "
            "(https://developer.1password.com/docs/cli/get-started/) or set secrets.onepassword.binary_path.")


def apply_onepassword_secrets(
    *, enabled: bool, env: Optional[dict[str, str]] = None, account: str = "",
    service_account_token_env: str = _DEFAULT_TOKEN_ENV, binary_path: str = "",
    override_existing: bool = True, cache_ttl_seconds: float = 300, home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve configured ``op://`` references and set them on ``os.environ``
    (``hermes secrets onepassword sync --apply``). Never raises. Refs already
    satisfied by the env (when ``override_existing`` is false) and the token var
    are skipped *before* fetching, so ``op`` never runs for a discarded value."""
    result = FetchResult()
    if not enabled:
        return result

    valid, warnings = _validate_references(env)
    result.warnings.extend(warnings)

    def _guarded(name: str) -> bool:
        """True when ``name`` must not be applied (token var or env already set)."""
        return name == service_account_token_env or (not override_existing and bool(os.environ.get(name)))

    result.skipped.extend(n for n in valid if _guarded(n))
    refs_to_fetch = {n: ref for n, ref in valid.items() if not _guarded(n)}
    if not refs_to_fetch:
        return result

    binary = find_op(binary_path)
    result.binary_path = binary
    if binary is None:
        result.error = _missing_binary_error(binary_path)
        return result

    try:
        secrets, fetch_warnings = fetch_onepassword_secrets(
            references=refs_to_fetch, account=account, token_env=service_account_token_env,
            binary=binary, cache_ttl_seconds=cache_ttl_seconds, home_path=home_path)
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(fetch_warnings)
    for name, value in secrets.items():
        if _guarded(name):  # defensive re-check: keys should already be ⊆ refs_to_fetch
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        os.environ[name] = value
        result.applied.append(name)
    return result


class OnePasswordSource(SecretSource):
    """1Password as a registered **mapped** source (explicit per-var bindings, so
    its claims outrank bulk sources on contested vars)."""

    name = "onepassword"
    label = "1Password"
    shape = "mapped"
    scheme = "op"
    token_env_key = "service_account_token_env"
    default_token_env = _DEFAULT_TOKEN_ENV
    # override_existing defaults True: an explicit VAR→op:// binding is the
    # strongest user intent; a stale .env line must not silently defeat it.
    override_existing_default = True
    _AUTH_HINT = ("Run `hermes secrets onepassword token` to paste a fresh service-account token "
                  "({token_env}), or `op signin` for an interactive session.")
    remediation_hints = {ErrorKind.AUTH_FAILED: _AUTH_HINT, ErrorKind.AUTH_EXPIRED: _AUTH_HINT,
                         ErrorKind.BINARY_MISSING: _MISSING_BINARY_HINT}

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "env": {"description": "Map of ENV_VAR -> op://vault/item/field reference", "default": {}},
            "account": {"description": "op --account shorthand (empty = default account)", "default": ""},
            "service_account_token_env": {"description": "Env var holding the service-account token "
                                                         "(unset = desktop/interactive session)",
                                          "default": _DEFAULT_TOKEN_ENV},
            "binary_path": {"description": "Pin the op binary (empty = resolve via PATH)", "default": ""},
            "cache_ttl_seconds": {"description": "Disk+memory cache TTL; 0 disables", "default": 300},
            "override_existing": {"description": "Resolved values overwrite .env/shell values", "default": True},
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        env_map = cfg.get("env")
        valid, warnings = _validate_references(env_map if isinstance(env_map, dict) else None)
        result.warnings.extend(warnings)
        if not valid:
            if not warnings:
                result.fail("secrets.onepassword.enabled is true but the env: map is "
                            "empty.  Add ENV_VAR: op://vault/item/field entries.", ErrorKind.NOT_CONFIGURED)
            return result

        binary_path = str(cfg.get("binary_path") or "")
        binary = find_op(binary_path)
        result.binary_path = binary
        if binary is None:
            return result.fail(_missing_binary_error(binary_path), ErrorKind.BINARY_MISSING)

        try:
            fetched = fetch_onepassword_result(
                references=valid, account=str(cfg.get("account") or ""), token_env=self.token_env(cfg),
                binary=binary, cache_ttl_seconds=coerce_float(cfg.get("cache_ttl_seconds", 300), 300.0),
                home_path=home_path)
        except RuntimeError as exc:
            return result.fail(str(exc), _classify_op_error(str(exc)))

        result.secrets = fetched.secrets
        result.warnings.extend(fetched.warnings)
        # A degraded pull must not present as a clean one. Carry the structured
        # list up so the orchestrator (and the operators reading cron output) can
        # see WHICH credential is missing and why, instead of inferring it from a
        # downstream "not found in environment".
        result.unresolved.update(fetched.unresolved)
        if result.unresolved and not result.error:
            result.error_kind = _classify_op_error(" ".join(result.unresolved.values()))
        return result


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches (after a token rotation, so the next
    startup resolves fresh instead of serving values cached under the old token)."""
    _STORE.clear(home_path)


_reset_cache_for_tests = clear_caches
