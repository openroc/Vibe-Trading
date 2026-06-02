"""TDX (通达信) MCP loader: A-share data via upgraded MCP server.

Connects to the TDX MCP server (default 192.168.3.2/192.168.3.53 on ports
3100/3101) over the streamable-HTTP transport (POST /mcp with JSON-RPC 2.0)
and exposes its 30+ tools for OHLCV data, sector/industry boards, stock
info, dividends, trading calendars, real-time snapshots, financial /
operating data, convertible bonds, IPO subscriptions, share capital, and
client actions (custom sectors, broadcast).

The transport is governed by MCP protocol version 2024-11-05: an
``initialize`` handshake is followed by ``tools/list`` and ``tools/call``,
with the session id carried in the ``Mcp-Session-Id`` header. The legacy
SSE endpoint (``/sse``) is no longer available on the upgraded server.

Only the ``fetch()`` method is called by the backtest runner. All other
methods are exposed for direct use in research workflows.

TDX period values: ``1d`` (daily) | ``1w`` (weekly) | ``1mon`` (monthly) |
``1y`` (yearly). Intraday bars are not supported.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import requests

from agent.backtest.loaders.base import validate_date_range
from agent.backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------
_TDX_HOSTS = os.getenv(
    "TDX_MCP_HOSTS",
    "192.168.3.2,192.168.3.53"
).split(",")

TDX_PORTS = [int(p) for p in os.getenv("TDX_MCP_PORTS", "3100,3101").split(",")]

TDX_TIMEOUT = int(os.getenv("TDX_MCP_TIMEOUT", "60"))

# TDX only supports 1d / 1w / 1mon / 1y
_PERIOD_MAP: Dict[str, str] = {
    "1D": "1d",
    "1W": "1w",
    "1M": "1mon",
    "1Y": "1y",
}

# Default dividend adjustment when fetching market data: "none" keeps raw
# prices, "front" applies forward adjustment, "back" applies backward
# adjustment.
_DIVIDEND_TYPE_MAP: Dict[str, str] = {
    "none": "none",
    "front": "front",
    "back": "back",
}

# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------

@register
class DataLoader:
    """TDX MCP-backed A-share OHLCV loader and research API.

    Uses the streamable-HTTP MCP transport (JSON-RPC 2.0 over POST /mcp) with
    a session id returned by the server on the ``initialize`` handshake.
    """

    name = "tdx"
    markets = {"a_share"}
    requires_auth = False

    def __init__(self) -> None:
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._session_id: Optional[str] = None
        self._request_id: int = 0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if any TDX MCP server responds to the initialize handshake."""
        for host in _TDX_HOSTS:
            for port in TDX_PORTS:
                if self._try_initialize(host, port):
                    return True
        logger.debug("No TDX MCP server reachable")
        return False

    # ------------------------------------------------------------------
    # Transport — streamable-HTTP / JSON-RPC 2.0
    # ------------------------------------------------------------------

    def _endpoint(self) -> str:
        return f"http://{self._host}:{self._port}/mcp"

    def _try_initialize(self, host: str, port: int) -> bool:
        """Open a session to host:port and send the ``initialize`` handshake."""
        try:
            self._host = host
            self._port = port
            self._session_id = None
            self._request_id = 0
            self._initialized = False

            response = self._post({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "vibe-trading-tdx-loader",
                        "version": "1.0.0",
                    },
                },
            })
            if response is None:
                return False
            if "error" in response:
                logger.debug("Initialize error from %s:%s: %s", host, port, response["error"])
                return False
            self._initialized = True
            logger.debug("Connected to TDX MCP at %s:%s (session=%s)", host, port, self._session_id)
            return True
        except Exception as exc:
            logger.debug("Failed to initialize TDX MCP at %s:%s: %s", host, port, exc)
            self._initialized = False
            self._session_id = None
            return False

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _post(self, payload: dict, *, timeout: Optional[int] = None) -> Optional[dict]:
        """POST a JSON-RPC 2.0 request to /mcp, return parsed response or None.

        Captures the ``Mcp-Session-Id`` header on every response (the server
        may rotate it). Returns ``None`` on any transport failure so callers
        can fall through cleanly.
        """
        if not self._host or not self._port:
            return None
        url = self._endpoint()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout if timeout is not None else TDX_TIMEOUT,
            )
        except Exception as exc:
            logger.debug("HTTP error talking to %s: %s", url, exc)
            return None

        new_sid = resp.headers.get("Mcp-Session-Id")
        if new_sid:
            self._session_id = new_sid

        if resp.status_code >= 400:
            logger.debug("HTTP %s from %s: %s", resp.status_code, url, resp.text[:200])
            return None

        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.debug("Non-JSON response from %s: %s", url, resp.text[:200])
            return None

    def _ensure_initialized(self) -> bool:
        """Make sure we hold an active MCP session; try every host:port combo."""
        if self._initialized and self._session_id is not None:
            return True
        for host in _TDX_HOSTS:
            for port in TDX_PORTS:
                if self._try_initialize(host, port):
                    return True
        return False

    def _call_tool(self, name: str, arguments: dict) -> Optional[Any]:
        """Call a TDX MCP tool and return the parsed payload (or None on failure).

        The transport returns ``{"jsonrpc":"2.0","id":N,"result":{...}}``.
        Tools differ on whether they nest their payload in the MCP standard
        ``content`` array (text item) or return the dict directly — we
        handle both transparently in :func:`_unwrap_result`.
        """
        if not self._ensure_initialized():
            return None

        response = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        })
        if response is None:
            return None
        if "error" in response:
            logger.debug("Tool %s error: %s", name, response["error"])
            return None
        return self._unwrap_result(response.get("result"))

    @staticmethod
    def _unwrap_result(result: Any) -> Any:
        """Normalize the heterogeneous ``result`` shapes returned by TDX tools.

        Three shapes are observed across the 30+ tools:

        1. ``{"content": [{"type": "text", "text": "<json string>"}]}`` — the
           MCP standard content wrapper; ``text`` is JSON-stringified.
        2. ``{"ErrorId": "0", "Data": {...}}`` — direct dict, no wrapper.
        3. A bare list (e.g. ``["600000.SH", "600519.SH"]``).

        All three collapse to the underlying Python object so callers can
        treat every tool response uniformly.
        """
        if not isinstance(result, dict):
            return result
        content = result.get("content")
        if isinstance(content, list) and content:
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    try:
                        return json.loads(item["text"])
                    except (json.JSONDecodeError, TypeError):
                        return item["text"]
            return None
        return result

    # ------------------------------------------------------------------
    # OHLCV fetch (called by backtest runner)
    # ------------------------------------------------------------------

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
        dividend_type: Literal["none", "front", "back"] = "none",
        fill_data: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch A-share OHLCV data via TDX MCP ``get_market_data``.

        TDX only supports daily-and-above periods (``1d``, ``1w``, ``1mon``,
        ``1y``). Intraday bars (``1m``–``1H``) fall back gracefully — the
        runner may substitute another source.

        Args:
            codes: Stock codes in full format, e.g. ``["600000.SH",
                "000001.SZ"]``.
            start_date: Start date, ``YYYY-MM-DD``.
            end_date: End date, ``YYYY-MM-DD``.
            interval: Bar size. Only ``"1D"`` is fully supported.
            fields: Optional list of field names to request
                (``Date``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume``,
                ``Amount``, ``ForwardFactor``, ``VolInStock``, ``Time``).
            dividend_type: Forward-adjustment mode — ``"none"`` (raw),
                ``"front"`` (forward), ``"back"`` (backward).
            fill_data: Whether to backfill missing bars with the last
                known close.

        Returns:
            Mapping ``code -> OHLCV DataFrame`` (index: ``trade_date``).
        """
        validate_date_range(start_date, end_date)

        period = _PERIOD_MAP.get(interval, "1d")
        tdx_start = start_date.replace("-", "")
        tdx_end = end_date.replace("-", "")

        result: Dict[str, pd.DataFrame] = {}

        try:
            args: Dict[str, Any] = {
                "stockList": codes,
                "period": period,
                "dividendType": _DIVIDEND_TYPE_MAP.get(dividend_type, "none"),
                "fillData": fill_data,
            }

            if period == "1d":
                args["startTime"] = tdx_start
                args["endTime"] = tdx_end

            if fields:
                args["fieldList"] = fields

            raw = self._call_tool("get_market_data", args)

            is_error = raw is None or (
                isinstance(raw, dict)
                and str(raw.get("ErrorId", "")) not in ("", "0")
            )
            if is_error:
                raw = self._call_tool(
                    "price_df",
                    {
                        "stockList": codes,
                        "period": period,
                        "startTime": tdx_start,
                        "endTime": tdx_end,
                    },
                )

            if raw is None:
                logger.warning("All TDX OHLCV calls failed")
                return result

            for code in codes:
                df = self._normalize_ohlcv(raw, code)
                if df is not None:
                    start_dt = pd.Timestamp(start_date)
                    end_dt = pd.Timestamp(end_date)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    if not df.empty:
                        result[code] = df

        except Exception as exc:
            logger.warning("TDX fetch failed: %s", exc)

        return result

    def _normalize_ohlcv(
        self, raw: Optional[dict], code: str
    ) -> Optional[pd.DataFrame]:
        """Parse TDX ``get_market_data`` or ``price_df`` response for one code.

        ``get_market_data`` format (upgraded server)::

            {
                "ErrorId": "0",
                "Data": {
                    "600000.SH": {
                        "ErrorId": "0",
                        "Date": ["20250506", ...],
                        "Time": ["09:30", ...],
                        "Open": ["11.00", ...],
                        "High": ["11.19", ...],
                        "Low": ["10.80", ...],
                        "Close": ["11.17", ...],
                        "Volume": ["97264624.00", ...],
                        "Amount": ["1085...", ...],
                        "VolInStock": [...],
                        "ForwardFactor": [...],
                    }
                }
            }

        ``price_df`` format (legacy fallback)::

            {
                "dates": ["20250506", ...],
                "data": {
                    "600000.SH": {
                        "Open": [...], "High": [...], ...
                    }
                }
            }
        """
        if not raw or not isinstance(raw, dict):
            return None

        data_for_code: Optional[dict] = None

        if "Data" in raw and isinstance(raw["Data"], dict):
            data_for_code = raw["Data"].get(code)
            if isinstance(data_for_code, dict) and "Data" in data_for_code:
                # Server may nest one level deeper on the upgraded protocol.
                data_for_code = data_for_code["Data"]

        if not data_for_code and "data" in raw:
            data_for_code = raw["data"].get(code)

        if not data_for_code or not isinstance(data_for_code, dict):
            return None

        def _arr(key: str) -> List[Any]:
            v = data_for_code.get(key)
            if v is None:
                v = data_for_code.get(key.lower())
            return v or []

        dates = _arr("Date")
        opens = _arr("Open")
        highs = _arr("High")
        lows = _arr("Low")
        closes = _arr("Close")
        volumes = _arr("Volume")
        amounts = _arr("Amount")
        forward_factors = _arr("ForwardFactor")
        vol_in_stock = _arr("VolInStock")

        if not dates:
            return None

        rows = []
        for i in range(len(dates)):
            date_str = str(dates[i])

            def _safe(idx: int, arr: List[Any], default: float) -> float:
                if idx >= len(arr):
                    return default
                try:
                    return float(arr[idx])
                except (ValueError, TypeError):
                    return default

            row = {
                "trade_date": date_str,
                "open": _safe(i, opens, 0.0),
                "high": _safe(i, highs, 0.0),
                "low": _safe(i, lows, 0.0),
                "close": _safe(i, closes, 0.0),
                "volume": _safe(i, volumes, 0.0),
                "amount": _safe(i, amounts, 0.0),
                "forward_factor": _safe(i, forward_factors, 1.0),
                "vol_in_stock": _safe(i, vol_in_stock, 0.0),
            }
            rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date").sort_index()

        for col in [
            "open", "high", "low", "close", "volume",
            "amount", "forward_factor", "vol_in_stock",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        ohlcv = df.dropna(subset=["open", "high", "low", "close"])
        return ohlcv if not ohlcv.empty else None

    # ------------------------------------------------------------------
    # Stock list
    # ------------------------------------------------------------------

    def get_stock_list(
        self,
        market: Literal[
            "0", "1", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
            "15", "16", "17", "18", "21", "22", "23", "24", "25", "26", "27",
            "28", "30", "31", "32", "33", "34", "35", "36", "49", "50", "51",
            "52", "53", "101", "102", "103", "91", "92"
        ] = "50",
        list_type: Literal[0, 1] = 1,
    ) -> pd.DataFrame:
        """Return stock/ETF/futures list.

        Args:
            market: Classification code. Notable values:

                - ``"50"`` — all A-shares
                - ``"31"`` — ETF funds
                - ``"32"`` — convertible bonds
                - ``"10"`` — all sector indices (板块指数)

                Full list: ``"0"`` (自选股), ``"1"`` (持仓股),
                ``"5"`` (所有A股), ``"10"`` (所有板块指数),
                ``"11"`` (缺省行业板块), ``"12"`` (概念板块),
                ``"13"`` (风格板块), ``"14"`` (地区板块),
                ``"15"`` (行业+概念), ``"16"`` (研究行业一级),
                ``"23"`` (沪深300), ``"24"`` (中证500),
                ``"25"`` (中证1000), ``"26"`` (国证2000),
                ``"31"`` (ETF), ``"32"`` (可转债), ``"50"`` (沪深A股),
                ``"51"`` (创业板), ``"52"`` (科创板), ``"53"`` (北交所),
                ``"101"`` (国内期货), ``"102"`` (港股), ``"103"`` (美股).

            list_type: 0 = codes only (``["000001.SZ"]``);
                1 = objects with code and name (``[{"Code":"...","Name":"..."}]``).

        Returns:
            DataFrame with columns ``stockCode`` and ``stockName``.
        """
        # Some clients pass list_type as snake_case; TDX accepts both.
        raw = self._call_tool(
            "get_stock_list", {"market": market, "listType": list_type}
        )
        return self._parse_list_payload(raw, list_type)

    def _parse_list_payload(
        self, raw: Any, list_type: int
    ) -> pd.DataFrame:
        """Parse stock/sector list payloads in either ErrorId-wrapped or bare form."""
        items: List[Any] = []
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            if "Data" in raw and isinstance(raw["Data"], list):
                items = raw["Data"]
            elif "Value" in raw and isinstance(raw["Value"], list):
                items = raw["Value"]
            else:
                return pd.DataFrame(columns=["stockCode", "stockName"])
        else:
            return pd.DataFrame(columns=["stockCode", "stockName"])

        if not items:
            return pd.DataFrame(columns=["stockCode", "stockName"])

        if list_type == 0 or all(isinstance(x, str) for x in items):
            return pd.DataFrame(
                {"stockCode": [c for c in items if c]},
                columns=["stockCode", "stockName"],
            )

        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append({
                "stockCode": item.get("Code", item.get("code", "")),
                "stockName": item.get("Name", item.get("name", "")),
            })
        return pd.DataFrame(rows, columns=["stockCode", "stockName"])

    # ------------------------------------------------------------------
    # Real-time snapshot
    # ------------------------------------------------------------------

    def get_market_snapshot(
        self,
        stock_code: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """Return a real-time market snapshot for a single stock or index.

        Key fields (always present unless filtered out):
        ``Now`` (current price), ``Open``, ``Max`` (high), ``Min`` (low),
        ``Volume`` (total volume), ``Amount`` (total turnover),
        ``LastClose`` (previous close), ``TickDiff`` (price change),
        ``ZAFPre`` (33-day return), ``Zjl`` (net main buy amount),
        ``Buyp/Buyv/Sellp/Sellv`` (5-level order book).

        Args:
            stock_code: Full stock code, e.g. ``"600000.SH"``.
            fields: Optional list of field names to return.

        Returns:
            Dict of field -> value, or None if unavailable.
        """
        args: Dict[str, Any] = {"stockCode": stock_code}
        if fields:
            args["fieldList"] = fields

        raw = self._call_tool("get_market_snapshot", args)
        if not isinstance(raw, dict):
            return None
        if "ErrorId" in raw and str(raw.get("ErrorId")) not in ("", "0"):
            return None
        return raw

    def get_market_snapshot_df(
        self,
        stock_code: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Same as ``get_market_snapshot`` but returns a single-row DataFrame."""
        snap = self.get_market_snapshot(stock_code, fields=fields)
        if not snap:
            return None
        df = pd.DataFrame([snap])
        df = df.drop(columns=["ErrorId"], errors="ignore")
        return df

    # ------------------------------------------------------------------
    # Trading calendar
    # ------------------------------------------------------------------

    def get_trading_calendar(
        self,
        market: Literal["SZ", "SH"] = "SH",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return trading dates for a market.

        Args:
            market: ``"SZ"`` (Shenzhen) or ``"SH"`` (Shanghai).
            start_date: Start date in ``YYYYMMDD`` or ``YYYY-MM-DD``.
            end_date: End date in ``YYYYMMDD`` or ``YYYY-MM-DD``.

        Returns:
            DataFrame with a single ``trade_date`` column (date objects).
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")

        raw = self._call_tool(
            "get_trading_calendar",
            {"market": market, "startTime": start, "endTime": end},
        )
        return self._parse_date_list(raw)

    def get_trading_dates(
        self,
        market: Literal["SZ", "SH"] = "SH",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return trading dates for a market (alternate TDX tool).

        Functionally equivalent to :func:`get_trading_calendar`; the upgraded
        server exposes both endpoints for client compatibility.
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")
        raw = self._call_tool(
            "get_trading_dates",
            {"market": market, "startTime": start, "endTime": end},
        )
        return self._parse_date_list(raw)

    @staticmethod
    def _parse_date_list(raw: Any) -> pd.DataFrame:
        if not isinstance(raw, list):
            return pd.DataFrame(columns=["trade_date"])
        dates = []
        for item in raw:
            s = str(item) if not isinstance(item, str) else item
            try:
                dates.append(pd.to_datetime(s, format="%Y%m%d").date())
            except Exception:
                continue
        return pd.DataFrame({"trade_date": dates})

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------

    def get_divid_factors(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return dividend and bonus data for a stock.

        Fields: ``Type`` (1=除权除息, 11=扩缩股, 15=重新调整),
        ``Bonus`` (红利), ``AlloPrice`` (配股价),
        ``ShareBonus`` (送股/扩缩股比例), ``Allotment`` (配股).

        Args:
            stock_code: Full stock code, e.g. ``"600000.SH"``.
            start_date: Start date in ``YYYYMMDD`` or ``YYYY-MM-DD``.
            end_date: End date in ``YYYYMMDD`` or ``YYYY-MM-DD``.

        Returns:
            DataFrame with columns ``trade_date``, ``Type``, ``Bonus``,
            ``AlloPrice``, ``ShareBonus``, ``Allotment``.
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")

        raw = self._call_tool(
            "get_divid_factors",
            {
                "stockCode": stock_code,
                "startTime": start,
                "endTime": end,
            },
        )

        items: List[Any] = []
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            if isinstance(raw.get("Data"), list):
                items = raw["Data"]
            elif isinstance(raw.get("Value"), list):
                items = raw["Value"]
            else:
                return pd.DataFrame(
                    columns=["trade_date", "Type", "Bonus", "AlloPrice",
                             "ShareBonus", "Allotment"]
                )
        else:
            return pd.DataFrame(
                columns=["trade_date", "Type", "Bonus", "AlloPrice",
                         "ShareBonus", "Allotment"]
            )

        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            date_str = str(item.get("Date", item.get("date", "")))
            try:
                trade_date = pd.to_datetime(date_str, format="%Y%m%d")
            except Exception:
                trade_date = pd.NaT

            # The upgraded server packs the per-date values into a parallel
            # ``Value`` array (one entry per stockList element). When called
            # with a single stock, take the first element.
            value_arr = item.get("Value")
            if isinstance(value_arr, list) and value_arr:
                v0 = value_arr[0]
                if isinstance(v0, dict):
                    row = {"trade_date": trade_date, **v0}
                else:
                    row = {
                        "trade_date": trade_date,
                        "Type": item.get("Type", ""),
                        "Bonus": item.get("Bonus", ""),
                        "AlloPrice": item.get("AlloPrice", ""),
                        "ShareBonus": item.get("ShareBonus", ""),
                        "Allotment": item.get("Allotment", ""),
                    }
                rows.append(row)
                continue

            rows.append({
                "trade_date": trade_date,
                "Type": item.get("Type", ""),
                "Bonus": item.get("Bonus", ""),
                "AlloPrice": item.get("AlloPrice", ""),
                "ShareBonus": item.get("ShareBonus", ""),
                "Allotment": item.get("Allotment", ""),
            })

        df = pd.DataFrame(rows)
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date")
            df = df.set_index("trade_date")
        return df

    # ------------------------------------------------------------------
    # Stock info
    # ------------------------------------------------------------------

    def get_stock_info(
        self,
        stock_code: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """Return fundamental info for a single stock.

        Key fields:
        ``Name`` (证券名称), ``BelongHS300``, ``IsSTGP``, ``IsQuitGP``,
        ``HSStockKind`` (0=指数,1=A股主板,2=北证A股,3=创业板,4=科创板,
        5=B股,6=债券,7=基金,8=权证,9=其它,10=非沪深京品种),
        ``ActiveCapital`` (流通股本), ``J_zgb`` (总股本),
        ``J_mgjzc`` (每股净资产), ``J_mgsy`` (每股收益),
        ``J_jyl`` (净资产收益率), ``rs_hyname`` (通达信行业),
        ``tdx_dyname`` (通达信地域), ``BelongRZRQ`` (融资融券标的),
        ``BelongHSGT`` (沪深股通).

        Args:
            stock_code: Full stock code, e.g. ``"600000.SH"``.
            fields: Optional list of field names to return.

        Returns:
            Dict of field -> value, or None if unavailable.
        """
        args: Dict[str, Any] = {"stockCode": stock_code}
        if fields:
            args["fieldList"] = fields

        raw = self._call_tool("get_stock_info", args)
        if not isinstance(raw, dict) or str(raw.get("ErrorId", "")) not in ("", "0", "None"):
            return None
        raw.pop("ErrorId", None)
        return raw

    def get_stock_info_df(
        self,
        stock_code: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Same as ``get_stock_info`` but returns a single-row DataFrame."""
        info = self.get_stock_info(stock_code, fields=fields)
        if not info:
            return None
        df = pd.DataFrame([info])
        return df

    # ------------------------------------------------------------------
    # Sector / industry board methods
    # ------------------------------------------------------------------

    def get_sector_list(
        self,
        list_type: Literal[0, 1] = 0,
    ) -> pd.DataFrame:
        """Return available sector / industry board names and codes.

        TDX returns two formats:
        - ``list_type=0``: plain code list ``["880081.SH", ...]`` — no names.
        - ``list_type=1``: objects with code and name
          ``[{"Code":"880081.SH","Name":"轮动趋势"}]``.

        When ``list_type=0`` this method resolves names by calling
        ``get_stock_info`` per code — this adds latency for large lists.

        Args:
            list_type: 0 = industry boards (code list, names resolved via
                ``get_stock_info``); 1 = concept/region boards (names
                included directly).

        Returns:
            DataFrame with columns ``blockCode`` and ``blockName``.
        """
        raw = self._call_tool("get_sector_list", {"listType": list_type})

        items: List[Any] = []
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("Data"), list):
            items = raw["Data"]
        else:
            return pd.DataFrame(columns=["blockCode", "blockName"])

        if not items:
            return pd.DataFrame(columns=["blockCode", "blockName"])

        if list_type == 1 or all(isinstance(x, dict) for x in items):
            rows = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "blockCode": item.get("Code", item.get("code", "")),
                    "blockName": item.get("Name", item.get("name", "")),
                })
            return pd.DataFrame(rows, columns=["blockCode", "blockName"])

        # list_type=0: industry boards — plain strings, need name resolution
        code_list = [c for c in items if isinstance(c, str) and c]
        name_map = self._resolve_sector_names(code_list)
        rows = [
            {"blockCode": c, "blockName": name_map.get(c, "")} for c in code_list
        ]
        df = pd.DataFrame(rows, columns=["blockCode", "blockName"])
        return df[df["blockName"] != ""]

    def _resolve_sector_names(self, codes: List[str]) -> Dict[str, str]:
        name_map: Dict[str, str] = {}
        for code in codes:
            info = self.get_stock_info(code)
            if info and info.get("Name"):
                name_map[code] = info["Name"]
        return name_map

    def get_stock_list_in_sector(
        self,
        block_code: str,
        block_type: Literal[0, 1] = 0,
        list_type: Literal[0, 1] = 1,
    ) -> pd.DataFrame:
        """Return constituent stocks of a sector / industry board.

        Args:
            block_code: Sector code (e.g. ``"881386.SH"``) or sector name
                (e.g. ``"全国性银行"``). ``get_sector_list()`` returns valid codes.
            block_type: 0 = sector index code or name; 1 = custom block short name
                (ZXG=自选股, TJG=临时条件股).
            list_type: 0 = codes only; 1 = codes and names.

        Returns:
            DataFrame with columns ``stockCode`` and ``stockName``.
        """
        raw = self._call_tool(
            "get_stock_list_in_sector",
            {
                "blockCode": block_code,
                "blockType": block_type,
                "listType": list_type,
            },
        )
        return self._parse_list_payload(raw, list_type)

    def get_stock_sectors(self, stock_code: str) -> List[Tuple[str, str, str]]:
        """Return all sector memberships for a stock.

        Args:
            stock_code: Full stock code, e.g. ``"600000.SH"``.

        Returns:
            List of ``(blockCode, blockName, blockType)`` tuples.
            ``blockType`` values: ``"行业"``, ``"地区"``, ``"概念"``, ``"风格"``.
        """
        raw = self._call_tool("get_relation", {"stockCode": stock_code})
        if not isinstance(raw, dict):
            return []

        items = raw.get("Value", raw.get("Data", []))
        if not isinstance(items, list):
            return []

        result: List[Tuple[str, str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            result.append((
                item.get("BlockCode", item.get("blockCode", "")),
                item.get("BlockName", item.get("blockName", "")),
                item.get("BlockType", item.get("blockType", "")),
            ))
        return result

    # ------------------------------------------------------------------
    # Market up/down count
    # ------------------------------------------------------------------

    def get_market_updown_count(self) -> Optional[dict]:
        """Return the market-wide up/down stock count.

        Returns:
            Dict with sub-dicts per market (``Shanghai``, ``Shenzhen``,
            ``Beijing``, ``Total``) and the per-market ``UpHome`` /
            ``DownHome`` (上涨家数 / 下跌家数) counts, or None if
            unavailable.
        """
        return self._call_tool("get_market_updown_count", {})

    # ------------------------------------------------------------------
    # More info (real-time metrics)
    # ------------------------------------------------------------------

    def get_more_info(self, stock_code: str) -> Optional[dict]:
        """Return extended real-time metrics for a stock.

        Key fields:
        ``ZAF`` (涨幅), ``Zjl`` (主买净额), ``Zjl_HB`` (主力净流入),
        ``PB_MRQ`` (市净率), ``DynaPE`` (动态市盈率), ``DYRatio`` (股息率),
        ``Zsz`` (总市值, 亿元), ``Ltsz`` (流通市值, 亿元),
        ``fHSL`` (换手率), ``fLianB`` (量比), ``Wtb`` (委比),
        ``MA5Value`` (5日均价), ``HisHigh`` / ``HisLow`` (52周高低),
        ``IsZCZGP`` (是否注册制A股), ``IsKzz`` (是否可转债),
        ``FreeLtgb`` (自由流通股本).

        Args:
            stock_code: Full stock code, e.g. ``"600000.SH"``.

        Returns:
            Dict of field -> value, or None if unavailable.
        """
        raw = self._call_tool("get_more_info", {"stockCode": stock_code})
        if not isinstance(raw, dict):
            return None
        # The upgraded server nests the metric dict under ``Value`` while
        # still echoing ``ErrorId`` at the top level.
        if "Value" in raw and isinstance(raw["Value"], dict):
            return raw["Value"]
        if str(raw.get("ErrorId", "")) not in ("", "0"):
            return None
        raw.pop("ErrorId", None)
        return raw

    # ------------------------------------------------------------------
    # Block / market / stock / sector value queries (upgraded payload shape)
    # ------------------------------------------------------------------

    def get_bkjy_value(
        self,
        block_codes: List[str],
        field_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return sector-level trading metrics over time.

        Key fields:
        ``BK5`` (市盈率TTM), ``BK6`` (市净率MRQ),
        ``BK7`` (市销率TTM), ``BK8`` (市现率TTM),
        ``BK9`` (涨跌数), ``BK10`` (板块总市值),
        ``BK11`` (板块流通市值), ``BK12`` (涨停数),
        ``BK15`` (融资融券), ``BK16`` (陆股通资金流入).

        The upgraded server strips the leading zero from the field name
        (``BK05`` → ``BK5``) and returns each field as a list of
        ``{Date, Value: [scalar_per_stock]}`` records.

        Args:
            block_codes: List of sector codes, e.g. ``["881386.SH"]``.
            field_list: List of field names, e.g. ``["BK5", "BK6", "BK9"]``.
            start_date: Start date in ``YYYYMMDD`` or ``YYYY-MM-DD``.
            end_date: End date in ``YYYYMMDD`` or ``YYYY-MM-DD``.

        Returns:
            DataFrame with ``trade_date`` index and one column per
            ``(blockCode, field)`` combination.
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")
        raw = self._call_tool(
            "get_bkjy_value",
            {
                "stockList": block_codes,
                "fieldList": field_list,
                "startTime": start,
                "endTime": end,
            },
        )
        return self._parse_value_payload(raw, block_codes, field_list)

    def get_bkjy_value_by_date(
        self,
        block_codes: List[str],
        field_list: List[str],
        year: int = 0,
        mmdd: int = 0,
    ) -> pd.DataFrame:
        """Return sector-level metrics anchored to a specific date.

        ``year=0``/``mmdd=0`` is the server's "latest" sentinel.
        """
        raw = self._call_tool(
            "get_bkjy_value_by_date",
            {
                "stockList": block_codes,
                "fieldList": field_list,
                "year": year,
                "mmdd": mmdd,
            },
        )
        return self._parse_value_payload(raw, block_codes, field_list)

    def get_scjy_value(
        self,
        field_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return market-wide trading metrics over time.

        Key fields:
        ``SC01`` (融资融券), ``SC02`` (陆股通资金流入),
        ``SC03`` (沪深京涨停股个数), ``SC04`` (沪深京跌停股个数),
        ``SC09`` (沪月新开A股账户), ``SC28`` (历史A股新高新低数),
        ``SC31`` (涨跌家数), ``SC33`` (市场总封单金额).
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")
        raw = self._call_tool(
            "get_scjy_value",
            {
                "fieldList": field_list,
                "startTime": start,
                "endTime": end,
            },
        )
        return self._parse_value_payload(raw, ["SC"], field_list)

    def get_scjy_value_by_date(
        self,
        field_list: List[str],
        year: int = 0,
        mmdd: int = 0,
    ) -> pd.DataFrame:
        """Return market-wide metrics anchored to a specific date."""
        raw = self._call_tool(
            "get_scjy_value_by_date",
            {
                "fieldList": field_list,
                "year": year,
                "mmdd": mmdd,
            },
        )
        return self._parse_value_payload(raw, ["SC"], field_list)

    def get_gpjy_value(
        self,
        stock_codes: List[str],
        field_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return per-stock operating metrics over time.

        Key fields: ``GP01``–``GPnn`` (见 TDX 经营数据字典).
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")
        raw = self._call_tool(
            "get_gpjy_value",
            {
                "stockList": stock_codes,
                "fieldList": field_list,
                "startTime": start,
                "endTime": end,
            },
        )
        return self._parse_value_payload(raw, stock_codes, field_list)

    def get_gpjy_value_by_date(
        self,
        stock_codes: List[str],
        field_list: List[str],
        year: int = 0,
        mmdd: int = 0,
    ) -> pd.DataFrame:
        """Return per-stock operating metrics anchored to a specific date."""
        raw = self._call_tool(
            "get_gpjy_value_by_date",
            {
                "stockList": stock_codes,
                "fieldList": field_list,
                "year": year,
                "mmdd": mmdd,
            },
        )
        return self._parse_value_payload(raw, stock_codes, field_list)

    def get_financial_data(
        self,
        stock_codes: List[str],
        field_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return professional financial data for one or more stocks.

        Key fields: ``FN001``–``FNnn`` (e.g. 净利润, 营收, ROE, 资产负债率).
        """
        start = (start_date or "").replace("-", "")
        end = (end_date or "").replace("-", "")
        raw = self._call_tool(
            "get_financial_data",
            {
                "stockList": stock_codes,
                "fieldList": field_list,
                "startTime": start,
                "endTime": end,
            },
        )
        return self._parse_value_payload(raw, stock_codes, field_list)

    def get_financial_data_by_date(
        self,
        stock_codes: List[str],
        field_list: List[str],
        year: int = 0,
        mmdd: int = 0,
    ) -> pd.DataFrame:
        """Return professional financial data anchored to a specific date."""
        raw = self._call_tool(
            "get_financial_data_by_date",
            {
                "stockList": stock_codes,
                "fieldList": field_list,
                "year": year,
                "mmdd": mmdd,
            },
        )
        return self._parse_value_payload(raw, stock_codes, field_list)

    def get_gp_one_data(
        self,
        stock_codes: List[str],
        field_list: List[str],
    ) -> pd.DataFrame:
        """Return consensus / one-off data (``一致预期``).

        Key fields: ``GO1`` (一致预期EPS), ``GO2`` (一致预期净利润).
        """
        raw = self._call_tool(
            "get_gp_one_data",
            {"stockList": stock_codes, "fieldList": field_list},
        )
        return self._parse_value_payload(raw, stock_codes, field_list)

    def get_cb_info(self, stock_code: str) -> Optional[pd.DataFrame]:
        """Return convertible bond info for a single bond code (e.g. ``113050.SH``)."""
        raw = self._call_tool("get_cb_info", {"stockCode": stock_code})
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("Data"), list):
            return pd.DataFrame(raw["Data"])
        if isinstance(raw.get("Value"), list):
            return pd.DataFrame(raw["Value"])
        return None

    def get_ipo_info(
        self,
        ipo_type: int = 0,
        ipo_date: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Return new-share subscription info.

        Args:
            ipo_type: ``0`` = all, ``1`` = today, ``2`` = this week, etc.
                Refer to the TDX schema for the full mapping.
            ipo_date: ``0`` for the default (latest), or a specific date in
                ``YYYYMMDD`` form.
        """
        raw = self._call_tool(
            "get_ipo_info", {"ipoType": ipo_type, "ipoDate": ipo_date}
        )
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("Data"), list):
            return pd.DataFrame(raw["Data"])
        if isinstance(raw.get("Value"), list):
            return pd.DataFrame(raw["Value"])
        return None

    def get_gb_info(
        self,
        stock_code: str,
        date_list: Optional[List[str]] = None,
        count: int = 1,
    ) -> Optional[pd.DataFrame]:
        """Return share-capital history (总股本/流通股本) for a stock.

        Args:
            stock_code: Full stock code, e.g. ``"600000.SH"``.
            date_list: Specific dates to query (``YYYYMMDD``); if omitted, the
                server uses ``count`` to page backward from today.
            count: How many records to return (1 = latest).
        """
        args: Dict[str, Any] = {"stockCode": stock_code, "count": count}
        if date_list:
            args["dateList"] = date_list
        raw = self._call_tool("get_gb_info", args)
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("Data"), list):
            return pd.DataFrame(raw["Data"])
        if isinstance(raw.get("Value"), list):
            return pd.DataFrame(raw["Value"])
        return None

    # ------------------------------------------------------------------
    # Cache / refresh
    # ------------------------------------------------------------------

    def refresh_cache(self) -> bool:
        """Force-refresh all market caches on the TDX server.

        Returns:
            True on accepted request, False if the call failed.
        """
        raw = self._call_tool("refresh_cache", {})
        return raw is not None

    def refresh_kline(
        self,
        stock_codes: List[str],
        period: str = "1d",
    ) -> bool:
        """Force-refresh the K-line cache for specific stocks."""
        raw = self._call_tool(
            "refresh_kline", {"stockList": stock_codes, "period": period}
        )
        return raw is not None

    def download_financial_data(
        self,
        stock_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """Trigger a server-side financial-data download (heavy operation)."""
        args: Dict[str, Any] = {"stockList": stock_codes}
        if start_date:
            args["startTime"] = start_date.replace("-", "")
        if end_date:
            args["endTime"] = end_date.replace("-", "")
        raw = self._call_tool("download_financial_data", args)
        return raw is not None

    # ------------------------------------------------------------------
    # Client-action / user-block methods
    # ------------------------------------------------------------------

    def send_user_block(
        self,
        stocks: List[str],
        block_code: str = "",
        show: bool = False,
    ) -> Optional[Any]:
        """Send a self-selected stock block to the connected TDX client."""
        return self._call_tool(
            "send_user_block",
            {"stocks": stocks, "blockCode": block_code, "show": show},
        )

    def send_message(self, message: str) -> Optional[Any]:
        """Broadcast a text message to the connected TDX client."""
        return self._call_tool("send_message", {"message": message})

    def create_sector(self, block_code: str, block_name: str) -> Optional[Any]:
        """Create a custom user-defined sector."""
        return self._call_tool(
            "create_sector", {"blockCode": block_code, "blockName": block_name}
        )

    def delete_sector(self, block_code: str) -> Optional[Any]:
        """Delete a custom user-defined sector."""
        return self._call_tool("delete_sector", {"blockCode": block_code})

    def rename_sector(self, block_code: str, block_name: str) -> Optional[Any]:
        """Rename a custom user-defined sector."""
        return self._call_tool(
            "rename_sector", {"blockCode": block_code, "blockName": block_name}
        )

    def clear_sector(self, block_code: str) -> Optional[Any]:
        """Empty a custom user-defined sector (remove all constituents)."""
        return self._call_tool("clear_sector", {"blockCode": block_code})

    # ------------------------------------------------------------------
    # Internal parser — handles the upgraded bkjy/scjy/gpjy/financial shape
    # ------------------------------------------------------------------

    def _parse_value_payload(
        self,
        raw: Any,
        codes: List[str],
        field_list: List[str],
    ) -> pd.DataFrame:
        """Parse the multi-shape payload returned by the *_value tool family.

        Two observed shapes:

        1. **Upgraded shape** — per-field record list::

               {"Data": {"<code>": {"<field>": [{"Date": "20250101",
                                                  "Value": [v1, v2, ...]},
                                                 ...]}}}

           Each ``Value`` element corresponds positionally to an entry in
           ``stockList``; with a single stock, the array is a one-element
           list containing the scalar value.

        2. **Legacy shape** — single Date array + per-field arrays::

               {"Data": {"<code>": {"Date": ["20250101", ...],
                                    "<field>": [v, v, ...]}}}

        This parser also handles the top-level ``Value`` wrapper that the
        upgraded server occasionally uses (``get_more_info`` is a similar
        case).
        """
        if raw is None:
            return pd.DataFrame(columns=field_list)

        # Error case
        if isinstance(raw, dict) and str(raw.get("ErrorId", "")) not in ("", "0"):
            return pd.DataFrame(columns=field_list)

        # Unwrap {Value: {...}} container (occurs in some *_value_by_date paths)
        if isinstance(raw, dict) and isinstance(raw.get("Value"), (dict, list)):
            return self._parse_value_payload(raw["Value"], codes, field_list)

        data_section: Any = None
        if isinstance(raw, dict):
            data_section = raw.get("Data", raw.get("data"))
        if not isinstance(data_section, dict):
            return pd.DataFrame(columns=field_list)

        rows_dict: Dict[Any, Dict[str, Any]] = {}

        for code in codes:
            block_data = data_section.get(code)
            if not isinstance(block_data, dict):
                # bkjy_value may also be keyed by field at the top level
                # (sector-wide single code), so look for the data block in
                # any single-value key.
                continue

            for field in field_list:
                value = block_data.get(field, block_data.get(field.lower()))

                # Upgraded shape: list[{Date, Value: [scalar, ...]}]
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    for record in value:
                        if not isinstance(record, dict):
                            continue
                        date_str = record.get("Date", record.get("date", ""))
                        try:
                            dt = pd.to_datetime(str(date_str), format="%Y%m%d")
                        except Exception:
                            continue
                        key = (dt, code)
                        row = rows_dict.setdefault(key, {"trade_date": dt})
                        scalar = self._first_scalar(record.get("Value"))
                        row[f"{code}_{field}"] = scalar
                    continue

                # Legacy shape: flat array aligned with a sibling Date array
                if isinstance(value, list):
                    dates = block_data.get("Date", block_data.get("date", []))
                    for i, date_str in enumerate(dates):
                        if i >= len(value):
                            break
                        try:
                            dt = pd.to_datetime(str(date_str), format="%Y%m%d")
                        except Exception:
                            continue
                        key = (dt, code)
                        row = rows_dict.setdefault(key, {"trade_date": dt})
                        row[f"{code}_{field}"] = self._coerce_float(value[i])
                    continue

        if not rows_dict:
            return pd.DataFrame(columns=field_list)

        df = pd.DataFrame(list(rows_dict.values()))
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date").set_index("trade_date")
        return df

    @staticmethod
    def _first_scalar(value: Any) -> Optional[float]:
        """Return the first scalar from a ``Value`` array, dropping noise."""
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                return None
            return DataLoader._coerce_float(value[0])
        return DataLoader._coerce_float(value)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
