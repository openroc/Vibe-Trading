"""Internal network (192.168.3.80) loader for A-share OHLCV + fundamentals.

Port 8000 — historical K-line data (daily/weekly/monthly):
  - /today1d/{market}.{code}.csv   → 前复权 OHLCV + volume + amount
  - /histday/{market}.{code}.csv   → 不复权 OHLCV + volume + amount
  - /factors/{market}.{code}.csv   → 复权因子表 (date, factor)

Port 8100 — real-time quotes and fundamentals:
  - /api/quote?symbols=sh.600519,sz.000001,...
  - /api/stock/info?symbol=sh.600519

Symbol format: sh.XXXXXX / sz.XXXXXX / bj.XXXXXX
  - sh = 上交所 (SSE), sz = 深交所 (SZSE), bj = 北交所 (BSE)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional

import pandas as pd

from backtest.loaders.base import NoAvailableSourceError, validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Host / port configuration
# ---------------------------------------------------------------------------

_INTERNAL_HOST = os.environ.get("INTERNAL_API_HOST", "192.168.3.80")
_INTERNAL_KLINE_PORT = int(os.environ.get("INTERNAL_KLINE_PORT", "8000"))
_INTERNAL_REALTIME_PORT = int(os.environ.get("INTERNAL_REALTIME_PORT", "8100"))

_KLINE_BASE = f"http://{_INTERNAL_HOST}:{_INTERNAL_KLINE_PORT}"
_REALTIME_BASE = f"http://{_INTERNAL_HOST}:{_INTERNAL_REALTIME_PORT}"

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_EXTRA_COLUMNS = ["amount", "turnover"]

# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

_Market = Literal["sh", "sz", "bj"]


def _to_market_code(project_code: str) -> tuple[_Market, str]:
    """Convert project code (e.g. 600519.SH) to (market, symbol)."""
    upper = project_code.upper()
    if upper.endswith(".SH"):
        return "sh", upper[:-3]
    if upper.endswith(".SZ"):
        return "sz", upper[:-3]
    if upper.endswith(".BJ"):
        return "bj", upper[:-3]
    if upper.startswith("60") or upper.startswith("688") or upper.startswith("51"):
        return "sh", upper
    if upper.startswith("00") or upper.startswith("30") or upper.startswith("002") or upper.startswith("003"):
        return "sz", upper
    if upper.startswith("8") or upper.startswith("4") or upper.startswith("92"):
        return "bj", upper
    return "sh", upper  # fallback


def _project_code(market: _Market, code: str) -> str:
    """Convert (market, code) back to project code (e.g. sh + 600519 → 600519.SH)."""
    suffix = {  # pragma: no cover
        "sh": "SH",
        "sz": "SZ",
        "bj": "BJ",
    }.get(market, "SH")
    return f"{code}.{suffix}"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@register
class DataLoader:
    """Internal-network A-share OHLCV + fundamentals loader."""

    name = "internal"
    markets = {"a_share"}
    requires_auth = False  # 内网无认证

    def is_available(self) -> bool:
        """Available when the internal server is reachable on port 8000."""
        try:
            import httpx  # noqa: F401
            r = httpx.get(
                f"{_KLINE_BASE}/today1d/sh.600519.csv",
                timeout=3.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def __init__(self) -> None:
        import httpx
        self._http = httpx.Client(timeout=30.0)

    def __del__(self) -> None:
        self._http.close()

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch A-share OHLCV via internal 8000/8100 API.

        Args:
            codes: Project symbols, e.g. ``600519.SH``.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            interval: Bar size. ``1D`` (default) fetches daily.
            fields: Extra columns. ``"amount"`` returns 成交额; fundamental
                fields (``"pe"``, ``"pb"``, ``"roe"``) trigger an 8100 info call.

        Returns:
            Mapping ``{symbol: DataFrame}``.
        """
        validate_date_range(start_date, end_date)

        if interval != "1D":
            logger.warning("internal loader: interval %s not yet supported, using 1D", interval)
            interval = "1D"

        result: Dict[str, pd.DataFrame] = {}
        need_amount = fields and "amount" in fields
        need_fundamentals = fields and any(f in (fields or []) for f in ("pe", "pb", "roe"))

        for code in codes:
            try:
                df = self._fetch_ohlcv(code, start_date, end_date, need_amount)
                if df is not None and not df.empty:
                    result[code] = df
            except Exception as exc:
                logger.warning("internal: failed for %s: %s", code, exc)

        # Merge fundamentals from 8100 if requested
        if need_fundamentals and result:
            self._merge_fundamentals(result, codes, start_date, end_date, fields or [])

        return result

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _fetch_ohlcv(
        self,
        code: str,
        start_date: str,
        end_date: str,
        with_amount: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV via today1d (前复权) endpoint.

        today1d CSV format (always has header row):
            datetime,open,close,high,low,vol,amount

        histday CSV format (always has header row):
            datetime,open,close,high,low,vol,amount,year,month,day,hour,minute

        Note: columns are open,close,high,low — we normalize to open,high,low,close.
        """
        market, sym = _to_market_code(code)
        url = f"{_KLINE_BASE}/today1d/{market}.{sym}.csv"

        try:
            r = self._http.get(url, timeout=30.0)
        except Exception as exc:
            raise NoAvailableSourceError(f"cannot connect to {_KLINE_BASE}: {exc}") from exc

        csv_text = r.text
        if r.status_code == 404 or not csv_text.strip():
            # Fall back to histday (不复权)
            url = f"{_KLINE_BASE}/histday/{market}.{sym}.csv"
            try:
                r = self._http.get(url, timeout=30.0)
            except Exception as exc:
                raise NoAvailableSourceError(f"cannot connect to {_KLINE_BASE}: {exc}") from exc
            csv_text = r.text

        if r.status_code != 200:
            logger.warning("internal: HTTP %s for %s", r.status_code, code)
            return None

        if not csv_text.strip():
            return None

        # today1d CSV 列名: datetime,open,close,high,low,vol,amount
        # 重命名为标准列名
        col_names = ["trade_date", "open", "close", "high", "low", "volume", "amount"]

        df = pd.read_csv(
            pd.io.common.StringIO(csv_text),
            header=0,  # always has header
            names=col_names,
            usecols=range(7),  # only the first 7 columns
        )

        df["trade_date"] = pd.to_datetime(
            df["trade_date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["trade_date"])
        df = df.set_index("trade_date").sort_index()

        # CSV order is open,close,high,low — normalize to open,high,low,close
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # Clip to requested window
        df = df.loc[start_date:end_date]

        # Ensure standard column order: open, high, low, close, volume[, amount]
        keep_cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        df = df[keep_cols].dropna(subset=["open", "high", "low", "close"])

        return df

    def _merge_fundamentals(
        self,
        result: Dict[str, pd.DataFrame],
        codes: List[str],
        start_date: str,
        end_date: str,
        fields: List[str],
    ) -> None:
        """Fetch PE/PB/ROE from 8100 /api/stock/info and attach to result DataFrames."""
        fundamental_fields = [f for f in fields if f in ("pe", "pb", "roe")]

        for code in codes:
            if code not in result:
                continue
            sym = _to_market_code(code)
            try:
                market, sym_code = _to_market_code(code)
                url = f"{_REALTIME_BASE}/api/stock/info?symbol={market}.{sym_code}"
                r = self._http.get(url, timeout=10.0)
                if r.status_code != 200:
                    continue
                data = r.json()
                info = data.get("data", {}) or {}
                # 8100 info returns latest value as of today — attach to all rows
                for f in fundamental_fields:
                    if f in info:
                        result[code][f] = float(info[f])
            except Exception as exc:
                logger.warning("internal: fund fetch failed for %s: %s", code, exc)