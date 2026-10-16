"""Hermetic tests for the 1Password (`op` CLI) secret source.

We never invoke the real ``op`` binary: ``subprocess.run`` is mocked so the
suite stays fast and offline-safe.  A live resolve is exercised manually via
``hermes secrets onepassword sync`` outside of pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import onepassword as op


@pytest.fixture(autouse=True)
def _reset_caches():
    op._reset_cache_for_tests()
    yield
    op._reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _clean_op_env(monkeypatch):
    """Start every test from a known 1Password auth state."""
    for key in list(os.environ):
        if key.startswith("OP_SESSION_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.delenv("OP_ACCOUNT", raising=False)
    monkeypatch.delenv("OP_CONNECT_HOST", raising=False)
    monkeypatch.delenv("OP_CONNECT_TOKEN", raising=False)
    yield


def _ok(value: str):
    return mock.Mock(returncode=0, stdout=value, stderr="")


def _err(code: int, stderr: str):
    return mock.Mock(returncode=code, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def test_validate_references_filters_bad_names_and_refs():
    refs = {
        "OPENAI_API_KEY": "op://Private/OpenAI/api key",
        "1BAD_NAME": "op://Private/x/y",          # bad env name
        "HAS SPACE": "op://Private/x/y",          # bad env name
        "NOT_A_REF": "https://example.com",        # not op://
        "WHITESPACE": "  op://Private/z/field  ",  # stripped + kept
    }
    valid, warnings = op._validate_references(refs)
    assert valid == {
        "OPENAI_API_KEY": "op://Private/OpenAI/api key",
        "WHITESPACE": "op://Private/z/field",
    }
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# fetch_onepassword_secrets
# ---------------------------------------------------------------------------


def test_fetch_happy_path(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    values = {
        "op://Private/OpenAI/api key": "sk-abc\n",
        "op://Private/Anthropic/credential": "sk-ant-xyz",
    }

    def fake_run(cmd, **kwargs):
        # argv list, never shell=True; reference passed after `--`.
        assert "--" in cmd
        ref = cmd[cmd.index("--") + 1]
        return _ok(values[ref])

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={
            "OPENAI_API_KEY": "op://Private/OpenAI/api key",
            "ANTHROPIC_API_KEY": "op://Private/Anthropic/credential",
        },
        binary=fake_op,
        use_cache=False,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-abc", "ANTHROPIC_API_KEY": "sk-ant-xyz"}
    assert warnings == []






def test_fetch_read_failure_becomes_warning(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(
        op.subprocess, "run", lambda *a, **k: _err(1, "\x1b[31m[ERROR] not signed in\x1b[0m")
    )

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )
    assert secrets == {}
    assert len(warnings) == 1
    # ANSI control sequences are fully scrubbed from the surfaced message.
    assert "\x1b" not in warnings[0]
    assert "[31m" not in warnings[0]
    assert "not signed in" in warnings[0]










# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_inprocess_cache_hit(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)
    for _ in range(2):
        op.fetch_onepassword_secrets(
            references={"K": "op://V/I/F"}, cache_ttl_seconds=60,
            binary=fake_op, home_path=tmp_path,
        )
    assert calls["n"] == 1  # second call served from L1 cache








def test_connect_credential_change_invalidates_cache(monkeypatch, tmp_path):
    """A different 1Password Connect identity must not reuse a cached value."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)

    monkeypatch.setenv("OP_CONNECT_HOST", "https://connect.example.com")
    monkeypatch.setenv("OP_CONNECT_TOKEN", "tokenA")
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    # Rotate the Connect token → new identity.
    monkeypatch.setenv("OP_CONNECT_TOKEN", "tokenB")
    op._CACHE.clear()
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    assert calls["n"] == 2  # cache key changed → refetch






# ---------------------------------------------------------------------------
# find_op
# ---------------------------------------------------------------------------


def test_find_op_pinned_path_not_on_path(tmp_path, monkeypatch):
    pinned = tmp_path / "op"
    pinned.write_text("")
    pinned.chmod(0o755)
    # PATH lookup must NOT be consulted when a binary_path is pinned.
    monkeypatch.setattr(op.shutil, "which", lambda name: "/usr/bin/op")
    assert op.find_op(str(pinned)) == pinned




def test_op_child_env_forwards_config_directory(monkeypatch):
    """The op child must retain an explicit 1Password config location."""
    monkeypatch.setenv("OP_CONFIG_DIR", "/tmp/op-config")
    monkeypatch.setenv("UNRELATED_PROVIDER_TOKEN", "must-not-leak")

    env = op._op_child_env("")

    assert env["OP_CONFIG_DIR"] == "/tmp/op-config"
    assert "UNRELATED_PROVIDER_TOKEN" not in env


# ---------------------------------------------------------------------------
# apply_onepassword_secrets
# ---------------------------------------------------------------------------


def test_apply_disabled_returns_empty():
    result = op.apply_onepassword_secrets(enabled=False, env={"K": "op://V/I/F"})
    assert result.ok
    assert not result.applied


def test_apply_missing_binary_sets_error(monkeypatch):
    monkeypatch.setattr(op, "find_op", lambda binary_path="": None)
    result = op.apply_onepassword_secrets(
        enabled=True, env={"K": "op://V/I/F"}
    )
    assert not result.ok
    assert "op CLI" in result.error


def test_apply_sets_env(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: _ok("resolved-val"))
    monkeypatch.delenv("MY_OP_KEY", raising=False)

    result = op.apply_onepassword_secrets(
        enabled=True, env={"MY_OP_KEY": "op://V/I/F"}, cache_ttl_seconds=0,
    )
    assert result.ok
    assert result.applied == ["MY_OP_KEY"]
    assert os.environ["MY_OP_KEY"] == "resolved-val"


def test_apply_skips_before_fetch_when_not_overriding(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setenv("MY_OP_KEY", "from-env")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("from-1password")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    result = op.apply_onepassword_secrets(
        enabled=True, env={"MY_OP_KEY": "op://V/I/F"},
        override_existing=False, cache_ttl_seconds=0,
    )
    assert "MY_OP_KEY" in result.skipped
    assert os.environ["MY_OP_KEY"] == "from-env"
    assert calls["n"] == 0  # never even called op for a value we'd discard


def test_apply_never_overrides_token_var(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "original")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("malicious")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"OP_SERVICE_ACCOUNT_TOKEN": "op://V/I/F"},
        override_existing=True, cache_ttl_seconds=0,
    )
    assert "OP_SERVICE_ACCOUNT_TOKEN" in result.skipped
    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "original"
    assert calls["n"] == 0

# ---------------------------------------------------------------------------
# Partial resolution must not present as a clean pull (regression)
#
# A vault/network blip fails every reference in one run. Before the fix the
# per-reference failures were collected as warnings only, so the fetch reported
# ok=True while supplying a credential set with entries missing. The consumer
# then failed later with "not found in environment", naming the symptom and
# hiding the cause.
# ---------------------------------------------------------------------------


def _flaky_read(fail_refs, error="could not read secret: Network: (failed to request)"):
    """subprocess.run stub: references in ``fail_refs`` fail, others succeed."""
    calls = {"n": 0}

    def _run(cmd, **kwargs):
        calls["n"] += 1
        ref = cmd[-1]
        if ref in fail_refs:
            return _err(1, f"[ERROR] could not read secret {ref}: {error}")
        return _ok(f"value-for-{ref}")

    return _run, calls


def test_partial_resolution_is_reported_as_partial_not_ok(monkeypatch, tmp_path):
    """A pull that resolved some but not all references is flagged, and names
    the missing credential."""
    refs = {"GOOD_TOKEN": "op://Vault/Item/good", "MISSING_TOKEN": "op://Vault/Item/missing"}
    run, _ = _flaky_read({"op://Vault/Item/missing"})
    monkeypatch.setattr(op.subprocess, "run", run)

    result = op.fetch_onepassword_result(
        references=refs, binary=Path("/usr/local/bin/op"), use_cache=False,
        home_path=tmp_path, retry_sleep=lambda _s: None)

    # The values that DID resolve are still supplied (a degraded pull stays useful).
    assert result.secrets == {"GOOD_TOKEN": "value-for-op://Vault/Item/good"}
    # ...but the pull is not presented as clean.
    assert result.unresolved, "a failed reference must be recorded as unresolved"
    assert "MISSING_TOKEN" in result.unresolved
    assert result.partial is True
    assert result.ok_fully is False


def test_fully_resolved_pull_reports_ok_fully(monkeypatch, tmp_path):
    """The control: a complete pull is not flagged (the check must not be
    vacuous — a fix that always reports degraded would also pass the test
    above)."""
    refs = {"A_TOKEN": "op://Vault/Item/a", "B_TOKEN": "op://Vault/Item/b"}
    run, _ = _flaky_read(set())
    monkeypatch.setattr(op.subprocess, "run", run)

    result = op.fetch_onepassword_result(
        references=refs, binary=Path("/usr/local/bin/op"), use_cache=False,
        home_path=tmp_path, retry_sleep=lambda _s: None)

    assert result.ok_fully is True
    assert result.partial is False
    assert result.unresolved == {}


def test_transient_network_failure_is_retried_and_recovers(monkeypatch, tmp_path):
    """A transport blip must not degrade the pull on its first occurrence."""
    refs = {"A_TOKEN": "op://Vault/Item/a"}
    calls = {"n": 0}

    def _run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _err(1, "[ERROR] error initializing client: Authentication: "
                            "(failed to auth.LookupAuth), Network: (failed to request)")
        return _ok("recovered-value")

    monkeypatch.setattr(op.subprocess, "run", _run)
    slept = []
    result = op.fetch_onepassword_result(
        references=refs, binary=Path("/usr/local/bin/op"), use_cache=False,
        home_path=tmp_path, retry_sleep=slept.append)

    assert calls["n"] == 2, "a transient failure must be retried"
    assert result.secrets == {"A_TOKEN": "recovered-value"}
    assert result.unresolved == {}
    assert slept, "retry must back off rather than spin"


def test_real_outage_error_text_is_classified_transient():
    """Pin the classification of the ACTUAL production error.

    The outage reports an auth failure and a network failure in one message, and
    the rules are first-match-wins. With `authentication` matched first the
    message classified as AUTH_FAILED (non-retryable), so the retry added above
    would never have fired on the real failure — the fix would have looked
    correct while doing nothing in production.
    """
    real = ("op read failed for 'op://Vault/Item/x': [ERROR] could not read secret "
            "'op://Vault/Item/x': error initializing client: Authentication: "
            "(failed to auth.LookupAuth), Network: (failed to request)")
    kind = op._classify_op_error(real)
    assert kind in op._TRANSIENT_KINDS, (
        f"the real outage message classified as {kind.value}, which is not retried — "
        "the retry would be dead code in production")
    # A genuinely rejected credential must still be non-retryable.
    assert op._classify_op_error("is not signed in") not in op._TRANSIENT_KINDS
    assert op._classify_op_error("401 Unauthorized") not in op._TRANSIENT_KINDS


def test_non_transient_failure_is_not_retried(monkeypatch, tmp_path):
    """A bad reference is deterministic: retrying wastes the wall-clock budget
    and delays a failure the user must fix."""
    refs = {"A_TOKEN": "op://Vault/Item/a"}
    calls = {"n": 0}

    def _run(cmd, **kwargs):
        calls["n"] += 1
        return _err(1, "[ERROR] could not read secret: is not signed in")

    monkeypatch.setattr(op.subprocess, "run", _run)
    result = op.fetch_onepassword_result(
        references=refs, binary=Path("/usr/local/bin/op"), use_cache=False,
        home_path=tmp_path, retry_sleep=lambda _s: None)

    assert calls["n"] == 1, "an auth failure must fail fast, not retry"
    assert "A_TOKEN" in result.unresolved


def test_degraded_pull_is_not_cached(monkeypatch, tmp_path):
    """A degraded pull must not be frozen for the TTL: caching it would serve a
    missing credential for the whole window even after the vault recovers."""
    refs = {"GOOD_TOKEN": "op://Vault/Item/good", "MISSING_TOKEN": "op://Vault/Item/missing"}
    run, _ = _flaky_read({"op://Vault/Item/missing"})
    monkeypatch.setattr(op.subprocess, "run", run)

    op.fetch_onepassword_result(
        references=refs, binary=Path("/usr/local/bin/op"), use_cache=True,
        cache_ttl_seconds=300, home_path=tmp_path, retry_sleep=lambda _s: None)

    # A second call must re-attempt rather than serve a cached degraded set.
    calls = {"n": 0}
    def _count(cmd, **kwargs):
        calls["n"] += 1
        return _ok("v")
    monkeypatch.setattr(op.subprocess, "run", _count)
    op.fetch_onepassword_result(
        references=refs, binary=Path("/usr/local/bin/op"), use_cache=True,
        cache_ttl_seconds=300, home_path=tmp_path, retry_sleep=lambda _s: None)
    assert calls["n"] > 0, "a degraded pull must not be served from cache"


# ---------------------------------------------------------------------------
# Wiring: the SOURCE the orchestrator actually calls must carry the signal.
# Testing fetch_onepassword_result alone would stay green even if
# OnePasswordSource.fetch dropped the field on the floor.
# ---------------------------------------------------------------------------


def test_source_fetch_surfaces_unresolved(monkeypatch, tmp_path):
    run, _ = _flaky_read({"op://Vault/Item/missing"})
    monkeypatch.setattr(op.subprocess, "run", run)

    src = op.OnePasswordSource()
    cfg = {"enabled": True, "cache_ttl_seconds": 0,
           "env": {"GOOD_TOKEN": "op://Vault/Item/good",
                   "MISSING_TOKEN": "op://Vault/Item/missing"}}
    result = src.fetch(cfg, tmp_path)

    assert result.secrets, "resolved values must still be supplied"
    assert "MISSING_TOKEN" in result.unresolved, (
        "OnePasswordSource.fetch must propagate unresolved references; "
        "dropping them here makes the whole fix decorative")
    assert result.partial is True
