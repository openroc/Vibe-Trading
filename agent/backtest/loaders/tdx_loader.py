"""TDX (通达信) MCP loader: A-share data via MCP server.

Connects to the TDX MCP server (192.168.3.2:3100/3101, 192.168.3.53:3100/3101)
and exposes its tools for OHLCV data, sector/industry boards, stock info,
dividends, trading calendars, and real-time snapshots.

Only the ``fetch()`` method is called by the backtest runner. All other methods
are exposed for direct use in research workflows.

TDX period values: ``1d`` (daily) | ``1w`` (weekly) | ``1mon`` (monthly) |
``1y`` (yearly). Intraday bars are not supported.
"""

from __future__ import annotations

from datetime import timedelta
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

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

# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------

@register
class DataLoader:
    """TDX MCP-backed A-share OHLCV loader and research API."""

    name = "tdx"
    markets = {"a_share"}
    requires_auth = False

    def __init__(self) -> None:
        self._session: Optional[Any] = None
        self._streams: Optional[Tuple[Any, Any]] = None
        self._host: Optional[str] = None
        self._port: Optional[int] = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if any TDX MCP server responds on the /sse endpoint."""
        try:
            import mcp  # noqa: F401
        except ImportError:
            logger.debug("mcp package not installed")
            return False
        # 直接尝试完整连接
        for host in _TDX_HOSTS:
            for port in TDX_PORTS:
                if self._try_connect(host, port):
                    self._cleanup()
                    return True
        logger.debug("No TDX MCP server reachable")
        return False
        logger.debug("No TDX MCP server reachable")
        return False

    # ------------------------------------------------------------------
    # MCP session management
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Establish connection to TDX MCP, trying all host:port combos."""
        if self._session is not None:
            return True

        for host in _TDX_HOSTS:
            for port in TDX_PORTS:
                if self._try_connect(host, port):
                    return True

        logger.warning("All TDX MCP servers unavailable")
        return False

    def _try_connect(self, host: str, port: int) -> bool:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            loop = asyncio.get_event_loop()
        except Exception as exc:
            logger.debug("Failed to get event loop: %s", exc)
            return False

        try:
            async def _async_connect():
                streams = sse_client(
                    f"http://{host}:{port}/sse", timeout=TDX_TIMEOUT
                )
                read_stream, write_stream = await streams.__aenter__()
                session = ClientSession(read_stream, write_stream, timedelta(seconds=TDX_TIMEOUT))
                await session.__aenter__()
                await session.initialize()
                return streams, session

            self._streams, self._session = loop.run_until_complete(
                _async_connect()
            )
            self._host = host
            self._port = port
            logger.debug("Connected to TDX MCP at %s:%s", host, port)
            return True
        except Exception as exc:
            logger.debug("Failed to connect to TDX MCP at %s:%s: %s", host, port, exc)
            self._session = None
            self._streams = None
            return False

    def _cleanup(self) -> None:
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            return

        try:
            async def _async_cleanup():
                if self._session is not None:
                    await self._session.__aexit__(None, None, None)
                if self._streams is not None:
                    await self._streams.__aexit__(None, None, None)

            loop.run_until_complete(_async_cleanup())
        except Exception:
            pass
        finally:
            self._session = None
            self._streams = None

    def _call_tool(self, name: str, arguments: dict) -> Optional[Any]:
        """Call a TDX MCP tool and return the parsed JSON result."""
        if not self._connect():
            return None

        try:
            loop = asyncio.get_event_loop()

            async def _async_call():
                result = await self._session.call_tool(name, arguments)
                if result and hasattr(result, "content"):
                    for item in result.content:
                        if (
                            hasattr(item, "type")
                            and item.type == "text"
                            and hasattr(item, "text")
                        ):
                            try:
                                return json.loads(item.text)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Invalid JSON from TDX tool %s", name
                                )
                                return None
                return None

            return loop.run_until_complete(_async_call())

        except Exception as exc:
            logger.warning("TDX tool %s failed: %s", name, exc)
            self._cleanup()
            return None

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
                ``Amount``, ``ForwardFactor``, ``VolInStock``).

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
            }

            if period == "1d":
                args["startTime"] = tdx_start
                args["endTime"] = tdx_end

            if fields:
                args["fieldList"] = fields

            # Try get_market_data; fall back to price_df on error or empty response.
            raw = self._call_tool("get_market_data", args)

            # ErrorId "0" = success; treat non-zero int or non-string-0 as error.
            # Empty/None response also triggers fallback.
            is_error = raw is None or (
                isinstance(raw, dict)
                and raw.get("ErrorId") not in (None, "0", 0)
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
        finally:
            self._cleanup()

        return result

    def _normalize_ohlcv(
        self, raw: Optional[dict], code: str
    ) -> Optional[pd.DataFrame]:
        """Parse TDX ``get_market_data`` or ``price_df`` response for one code.

        ``get_market_data`` format::

            {
                "ErrorId": "0",
                "Data": {
                    "600000.SH": {
                        "Date": ["20250506", ...],
                        "Open": ["11.00", ...],
                        "High": ["11.19", ...],
                        "Low": ["10.80", ...],
                        "Close": ["11.17", ...],
                        "Volume": ["97264624.00", ...],
                        "Amount": ["1085...", ...],
                    }
                }
            }

        ``price_df`` format::

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

        # get_market_data
        if "Data" in raw and isinstance(raw["Data"], dict):
            data_for_code = raw["Data"].get(code)

        # price_df
        if not data_for_code and "data" in raw:
            data_for_code = raw["data"].get(code)

        if not data_for_code or not isinstance(data_for_code, dict):
            return None

        # Support both PascalCase and lowercase keys
        def _arr(key: str) -> List[Any]:
            return (
                data_for_code.get(key, [])
                or data_for_code.get(key.lower(), [])
            )

        dates = _arr("Date")
        opens = _arr("Open")
        highs = _arr("High")
        lows = _arr("Low")
        closes = _arr("Close")
        volumes = _arr("Volume")
        amounts = _arr("Amount")

        if not dates:
            return None

        rows = []
        for i in range(len(dates)):
            try:
                date_str = str(dates[i])
                rows.append([
                    date_str,
                    float(opens[i]) if i < len(opens) else 0.0,
                    float(highs[i]) if i < len(highs) else 0.0,
                    float(lows[i]) if i < len(lows) else 0.0,
                    float(closes[i]) if i < len(closes) else 0.0,
                    float(volumes[i]) if i < len(volumes) else 0.0,
                    float(amounts[i]) if amounts and i < len(amounts) else 0.0,
                ])
            except (ValueError, TypeError, IndexError):
                continue

        if not rows:
            return None

        df = pd.DataFrame(
            rows, columns=["trade_date", "open", "high", "low", "close", "volume", "amount"]
        )
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date").sort_index()

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        ohlcv = df[["open", "high", "low", "close", "volume", "amount"]].dropna(
            subset=["open", "high", "low", "close"]
        )
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
        raw = self._call_tool(
            "get_stock_list", {"market": market, "listType": list_type}
        )
        if not isinstance(raw, list) or not raw:
            return pd.DataFrame(columns=["stockCode", "stockName"])

        if list_type == 0:
            return pd.DataFrame(
                {"stockCode": [c for c in raw if c]},
                columns=["stockCode", "stockName"],
            )

        rows = []
        for item in raw:
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

        # Strip ErrorId wrapper if present
        if "ErrorId" in raw and raw["ErrorId"] != "0":
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
        # Drop the ErrorId column if present
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
        if not isinstance(raw, list):
            return pd.DataFrame(
                columns=["trade_date", "Type", "Bonus", "AlloPrice",
                         "ShareBonus", "Allotment"]
            )

        rows = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            date_str = str(item.get("Date", item.get("date", "")))
            try:
                trade_date = pd.to_datetime(date_str, format="%Y%m%d")
            except Exception:
                trade_date = pd.NaT

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
        if not isinstance(raw, dict) or raw.get("ErrorId") not in (None, "0"):
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
        if not isinstance(raw, list) or not raw:
            return pd.DataFrame(columns=["blockCode", "blockName"])

        if list_type == 1:
            # list_type=1 already has names
            rows = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "blockCode": item.get("Code", item.get("code", "")),
                    "blockName": item.get("Name", item.get("name", "")),
                })
            return pd.DataFrame(rows, columns=["blockCode", "blockName"])

        # list_type=0: industry boards — plain strings, need name resolution
        code_list = [c for c in raw if c]
        name_map = self._resolve_sector_names(code_list)

        rows = []
        for code in code_list:
            rows.append({
                "blockCode": code,
                "blockName": name_map.get(code, ""),
            })
        df = pd.DataFrame(rows, columns=["blockCode", "blockName"])
        return df[df["blockName"] != ""]

    def _resolve_sector_names(self, codes: List[str]) -> Dict[str, str]:
        """Fetch ``Name`` via ``get_stock_info`` for each sector code."""
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
        if not isinstance(raw, list) or not raw:
            return pd.DataFrame(columns=["stockCode", "stockName"])

        if list_type == 0:
            return pd.DataFrame(
                {"stockCode": [c for c in raw if c]},
                columns=["stockCode", "stockName"],
            )

        rows = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            rows.append({
                "stockCode": item.get("Code", item.get("code", "")),
                "stockName": item.get("Name", item.get("name", "")),
            })
        return pd.DataFrame(rows, columns=["stockCode", "stockName"])

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

        items = raw.get("Value", raw) if isinstance(raw, dict) else []
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
            Dict with fields such as ``UpHome`` (上涨家数) and
            ``DownHome`` (下跌家数) for indices, or None if unavailable.
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
        if not isinstance(raw, dict) or raw.get("ErrorId") not in (None, "0"):
            return None
        raw.pop("ErrorId", None)
        return raw

    # ------------------------------------------------------------------
    # Block-level trading data
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
        ``BK05`` (市盈率TTM), ``BK06`` (市净率MRQ),
        ``BK07`` (市销率TTM), ``BK08`` (市现率TTM),
        ``BK09`` (涨跌数), ``BK10`` (板块总市值),
        ``BK11`` (板块流通市值), ``BK12`` (涨停数),
        ``BK15`` (融资融券), ``BK16`` (陆股通资金流入).

        Args:
            block_codes: List of sector codes, e.g. ``["881386.SH"]``.
            field_list: List of field names, e.g. ``["BK05", "BK06", "BK09"]``.
            start_date: Start date in ``YYYYMMDD`` or ``YYYY-MM-DD``.
            end_date: End date in ``YYYYMMDD`` or ``YYYY-MM-DD``.

        Returns:
            DataFrame with multi-index (date, blockCode) and one column per
            field name.
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
        return self._parse_market_data(raw, block_codes, field_list)

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

        Args:
            field_list: List of field names, e.g. ``["SC01", "SC02"]``.
            start_date: Start date in ``YYYYMMDD`` or ``YYYY-MM-DD``.
            end_date: End date in ``YYYYMMDD`` or ``YYYY-MM-DD``.

        Returns:
            DataFrame with ``trade_date`` index and one column per field.
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
        return self._parse_market_data(raw, ["SC"], field_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_market_data(
        self,
        raw: Optional[Any],
        codes: List[str],
        field_list: List[str],
    ) -> pd.DataFrame:
        """Parse ``get_bkjy_value`` / ``get_scjy_value`` responses.

        Both return the same dict-of-dicts format as ``get_market_data``:
        ``{"ErrorId":"0","Data":{"880001.SH":{"Date":[...],"BK05":[...]}}}``.
        """
        if not isinstance(raw, dict):
            return pd.DataFrame(columns=field_list)

        error_id = raw.get("ErrorId", "0")
        if error_id != "0":
            return pd.DataFrame(columns=field_list)

        data = raw.get("Data", {})
        if not isinstance(data, dict):
            return pd.DataFrame(columns=field_list)

        rows_dict: Dict[str, dict] = {}

        for code in codes:
            block_data = data.get(code)
            if not isinstance(block_data, dict):
                continue

            dates = block_data.get("Date", [])
            n = len(dates)

            for i, date_str in enumerate(dates):
                if i >= n:
                    break
                try:
                    dt = pd.to_datetime(str(date_str), format="%Y%m%d")
                except Exception:
                    continue

                row_key = str(dt.date())
                if row_key not in rows_dict:
                    rows_dict[row_key] = {"trade_date": dt}

                for field in field_list:
                    arr = block_data.get(field, [])
                    rows_dict[row_key][f"{code}_{field}"] = (
                        float(arr[i]) if i < len(arr) else None
                    )

        if not rows_dict:
            return pd.DataFrame(columns=field_list)

        df = pd.DataFrame(list(rows_dict.values()))
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date").set_index("trade_date")
        return df
