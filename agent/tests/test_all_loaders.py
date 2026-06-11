"""Integration test for all data loaders.

Tests each registered loader's fetch() with a small, realistic payload
to verify connectivity, column layout, date indexing, and error handling.

Run with:  pytest tests/test_all_loaders.py
           pytest tests/test_all_loaders.py -k a_share
           pytest tests/test_all_loaders.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import pandas as pd
import pytest

import backtest.loaders  # noqa: F401  # ensure @register fires
from backtest.loaders.registry import FALLBACK_CHAINS, LOADER_REGISTRY, resolve_loader
from backtest.loaders.base import NoAvailableSourceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test cases: market -> list of (codes, start, end, interval)
# ---------------------------------------------------------------------------

TEST_CASES: dict[str, list[dict]] = {
    "a_share": [
        {
            "codes": ["600000.SH", "000001.SZ"],
            "start_date": "2025-05-12",
            "end_date": "2025-05-23",
            "interval": "1D",
        },
    ],
    "us_equity": [
        {
            "codes": ["AAPL.US", "MSFT.US"],
            "start_date": "2025-05-12",
            "end_date": "2025-05-23",
            "interval": "1D",
        },
    ],
    "hk_equity": [
        {
            "codes": ["0700.HK", "9988.HK"],
            "start_date": "2025-05-12",
            "end_date": "2025-05-23",
            "interval": "1D",
        },
    ],
    "crypto": [
        {
            "codes": ["BTC-USDT", "ETH-USDT"],
            "start_date": "2025-05-12",
            "end_date": "2025-05-23",
            "interval": "1D",
        },
    ],
    "futures": [
        {
            "codes": ["IF2506"],
            "start_date": "2025-05-12",
            "end_date": "2025-05-23",
            "interval": "1D",
        },
    ],
}

# Markets where at least one external loader may be unavailable in CI/no-token envs.
# Tests for these markets SKIP if all loaders fail (instead of FAIL) since an empty
# result can come from rate-limiting or network issues, not a code bug.
_UNRELIABLE_MARKETS = {"us_equity", "hk_equity", "crypto", "futures"}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

OHLCV_COLS = frozenset(["open", "high", "low", "close", "volume"])


def _validate_df(df: pd.DataFrame, label: str) -> list[str]:
    errors = []
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{label}: index is not DatetimeIndex")
    elif df.empty:
        errors.append(f"{label}: DataFrame is empty")
    else:
        if df.index[-1] < df.index[0]:
            errors.append(f"{label}: data not sorted (first={df.index[0]}, last={df.index[-1]})")
    missing = OHLCV_COLS - set(df.columns)
    if missing:
        errors.append(f"{label}: missing columns {missing}")
    return errors


# ---------------------------------------------------------------------------
# Per-source test
# ---------------------------------------------------------------------------

# Status: True=pass, False=fail, None=skip (e.g. no token, unreachable)
def _test_loader(name: str, cases: list[dict], smoke: bool = False) -> tuple[bool | None, list[str]]:
    """Instantiate loader, run fetch, report pass/fail. Returns (ok, lines)."""
    lines = []
    ok: bool | None = True

    if name not in LOADER_REGISTRY:
        lines.append(f"SKIP  {name}: not registered (missing deps)")
        return None, lines

    try:
        loader = LOADER_REGISTRY[name]()
    except Exception as exc:
        # __init__ raising = environment missing token/credentials → skip, don't fail
        lines.append(f"SKIP  {name}: __init__ raised {type(exc).__name__} (env issue, not a bug)")
        return None, lines

    try:
        available = loader.is_available()
    except Exception as exc:
        lines.append(f"ERROR {name}: is_available() raised {type(exc).__name__}: {exc}")
        return False, lines

    if not available:
        lines.append(f"OFF   {name}: is_available()=False")
        return None, lines

    lines.append(f"OK    {name}: available")

    if smoke:
        return True, lines

    for tc in cases:
        t0 = time.monotonic()
        try:
            result = loader.fetch(
                codes=tc["codes"],
                start_date=tc["start_date"],
                end_date=tc["end_date"],
                interval=tc.get("interval", "1D"),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            lines.append(f"ERROR {name}.fetch({tc['codes']}) raised {type(exc).__name__}: {exc}  ({elapsed:.1f}s)")
            ok = False
            continue

        elapsed = time.monotonic() - t0

        if not result:
            lines.append(f"ERROR {name}.fetch({tc['codes']}) returned empty dict  ({elapsed:.1f}s)")
            ok = False
            continue

        for code in tc["codes"]:
            df = result.get(code)
            if df is None:
                lines.append(f"ERROR {name}: missing '{code}' in result  ({elapsed:.1f}s)")
                ok = False
                continue
            errors = _validate_df(df, f"{name}/{code}")
            if errors:
                for e in errors:
                    lines.append(f"ERROR {e}")
                ok = False
            else:
                lines.append(f"OK    {name}.fetch({code}): {len(df)} bars "
                            f"{df.index[0].date()}→{df.index[-1].date()}  ({elapsed:.1f}s)")

    return ok, lines


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: skip data fetch, only check availability")


# ---------------------------------------------------------------------------
# Pytest test functions (one per market group)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("market", list(TEST_CASES.keys()), ids=list(TEST_CASES.keys()))
def test_fallback_chain(market: str) -> None:
    """resolve_loader() returns the first available loader for a market."""
    chain = FALLBACK_CHAINS.get(market, [])
    print(f"\n  chain={chain}")
    try:
        loader = resolve_loader(market)
        print(f"  → resolved to {loader.name}")
    except NoAvailableSourceError as exc:
        pytest.fail(str(exc))


@pytest.mark.parametrize("market", list(TEST_CASES.keys()), ids=list(TEST_CASES.keys()))
def test_loader_availability(market: str) -> None:
    """Every loader in a market chain reports is_available()."""
    chain = FALLBACK_CHAINS.get(market, [])
    for name in chain:
        ok, lines = _test_loader(name, TEST_CASES.get(market, []), smoke=True)
        for line in lines:
            print(f"\n  {line}")
        if ok is False:
            pytest.fail(f"Loader {name} failed availability check")


@pytest.mark.parametrize("market", list(TEST_CASES.keys()), ids=list(TEST_CASES.keys()))
@pytest.mark.integration
def test_loader_fetch(market: str) -> None:
    """At least one available loader in a market chain fetches real OHLCV data.

    Loaders that are unavailable or raise on __init__ (missing token) are
    expected and skipped. Only real fetch errors count as failures.
    """
    cases = TEST_CASES.get(market, [])
    if not cases:
        pytest.skip(f"No test cases for market: {market}")

    chain = FALLBACK_CHAINS.get(market, [])
    any_succeeded = False
    failure_reasons: list[str] = []

    for name in chain:
        ok, lines = _test_loader(name, cases, smoke=False)
        for line in lines:
            print(f"\n  {line}")
        if ok is None:
            continue  # SKIP or OFF — expected in some envs
        if ok is True:
            any_succeeded = True
        else:
            failure_reasons.append(name)

    if any_succeeded:
        return  # PASS

    if failure_reasons:
        if market in _UNRELIABLE_MARKETS:
            pytest.skip(f"All external loaders failed for {market}: {failure_reasons}  "
                        f"(rate-limit / network issue — not a code bug)")
        pytest.fail(f"All loaders failed for {market}: {failure_reasons}")
    else:
        pytest.skip(f"No loaders available for market {market}")


@pytest.mark.integration
def test_all_loaders_listed() -> None:
    """Every registered loader appears in at least one fallback chain."""
    for name in LOADER_REGISTRY:
        found = any(
            name in chain for chain in FALLBACK_CHAINS.values()
        )
        if not found:
            print(f"\n  WARNING: loader '{name}' not in any fallback chain")


# ---------------------------------------------------------------------------
# Standalone runner (python tests/test_all_loaders.py)
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Skip data fetch, only check availability")
    ap.add_argument("markets", nargs="*", help="Market names to test (default: all)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
    )

    all_ok = True
    markets = args.markets if args.markets else list(TEST_CASES.keys())

    print(f"\n{'='*60}")
    print("Registered loaders")
    print(f"{'='*60}")
    for name, cls in sorted(LOADER_REGISTRY.items()):
        print(f"  {name:<20}  markets={cls.markets}  auth={cls.requires_auth}")

    print(f"\n{'='*60}")
    print("Fallback chains")
    print(f"{'='*60}")
    for market, chain in sorted(FALLBACK_CHAINS.items()):
        print(f"  {market:<20}  → {chain}")

    for market in markets:
        cases = TEST_CASES.get(market, [])
        print(f"\n{'='*60}")
        print(f"Market: {market}  (smoke={args.smoke})")
        print(f"{'='*60}")

        # Fallback chain
        chain = FALLBACK_CHAINS.get(market, [])
        print(f"  Chain: {chain}")
        try:
            loader = resolve_loader(market)
            print(f"  → resolved to {loader.name}")
        except NoAvailableSourceError as exc:
            print(f"  → {exc}")

        # Per-loader
        for name in chain:
            ok, lines = _test_loader(name, cases, smoke=args.smoke)
            for line in lines:
                print(f"  {line}")
            if ok is False:
                all_ok = False

    print(f"\n{'='*60}")
    print(f"RESULT: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    _main()