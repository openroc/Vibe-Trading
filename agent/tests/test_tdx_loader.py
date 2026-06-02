"""Tests for TDX MCP loader — all external calls are mocked.

Transport layer is the upgraded streamable-HTTP / JSON-RPC 2.0 endpoint
(``POST /mcp``); ``requests.post`` is mocked at the boundary and the
``_call_tool`` shim is monkey-patched for payload-level tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure the loader module is registered as an attribute of
# ``backtest.loaders`` before any ``patch("backtest.loaders.tdx_loader.requests.post")``
# runs — otherwise the dotted-name resolver in ``patch`` can't find it.
from backtest.loaders import tdx_loader as _tdx_loader  # noqa: F401

# ------------------------------------------------------------------
# TDX response fixtures (shapes observed on the upgraded server)
# ------------------------------------------------------------------

_MOCK_KDATA = {
    "ErrorId": "0",
    "Data": {
        "600000.SH": {
            "ErrorId": "0",
            "Date": ["20250603", "20250604", "20250605"],
            "Open":  ["11.00", "11.20", "11.30"],
            "High":  ["11.19", "11.50", "11.60"],
            "Low":   ["10.80", "11.10", "11.20"],
            "Close": ["11.17", "11.45", "11.55"],
            "Volume":["97264624.00", "85000000.00", "91000000.00"],
            "Amount":["1085672345.67", "972345678.90", "1024567890.12"],
        }
    }
}

_MOCK_PRICE_DF = {
    "dates": ["20250603", "20250604", "20250605"],
    "data": {
        "600000.SH": {
            "Date":  ["20250603", "20250604", "20250605"],
            "Open":  ["11.00", "11.20", "11.30"],
            "High":  ["11.19", "11.50", "11.60"],
            "Low":   ["10.80", "11.10", "11.20"],
            "Close": ["11.17", "11.45", "11.55"],
            "Volume":["97264624.00", "85000000.00", "91000000.00"],
        }
    }
}

_MOCK_STOCK_LIST_LT0 = ["600000.SH", "600519.SH", "000001.SZ"]
_MOCK_STOCK_LIST_LT1 = [
    {"Code": "600000.SH", "Name": "浦发银行"},
    {"Code": "600519.SH", "Name": "贵州茅台"},
    {"Code": "000001.SZ", "Name": "平安银行"},
]

_MOCK_STOCK_LIST_IN_SECTOR_LT0 = ["600000.SH", "000001.SZ"]
_MOCK_STOCK_LIST_IN_SECTOR_LT1 = [
    {"Code": "600000.SH", "Name": "浦发银行"},
    {"Code": "000001.SZ", "Name": "平安银行"},
]

_MOCK_STOCK_INFO = {
    "ErrorId": "0",
    "Name": "浦发银行",
    "HSStockKind": "1",
    "rs_hyname": "银行",
    "BelongHS300": "1",
}

_MOCK_STOCK_INFO_LOWER = {
    "errorid": "0",
    "name": "浦发银行",
    "hsstockkind": "1",
}

_MOCK_SNAPSHOT = {
    "Now": "11.45",
    "Open": "11.20",
    "Max": "11.50",
    "Min": "11.11",
    "Volume": "85000000.00",
    "Amount": "972646240.00",
}

_MOCK_TRADING_CALENDAR = ["20250603", "20250604", "20250605"]
_MOCK_DIVID_FACTORS = [
    {
        "Date": "20250516",
        "Type": "1",
        "Bonus": "0.37",
        "ShareBonus": "0",
        "Allotment": "0",
        "AlloPrice": "0",
    }
]
# Upgraded bkjy payload: per-field record list, no leading-zero field names
_MOCK_MARKET_DATA_BKJY = {
    "ErrorId": "0",
    "Data": {
        "881386.SH": {
            "BK5": [
                {"Date": "20250603", "Value": ["6.50"]},
                {"Date": "20250604", "Value": ["6.55"]},
                {"Date": "20250605", "Value": ["6.60"]},
            ],
            "BK9": [
                {"Date": "20250603", "Value": ["12"]},
                {"Date": "20250604", "Value": ["15"]},
                {"Date": "20250605", "Value": ["10"]},
            ],
        }
    },
}
# Per-stock variant of the same shape — used by get_financial_data /
# get_gpjy_value which key on stock code, not block code.
_MOCK_MARKET_DATA_PER_STOCK = {
    "ErrorId": "0",
    "Data": {
        "600000.SH": {
            "BK5": [
                {"Date": "20250603", "Value": ["1.20"]},
                {"Date": "20250604", "Value": ["1.25"]},
                {"Date": "20250605", "Value": ["1.30"]},
            ],
            "BK9": [
                {"Date": "20250603", "Value": ["42"]},
                {"Date": "20250604", "Value": ["45"]},
                {"Date": "20250605", "Value": ["38"]},
            ],
        }
    },
}
# Legacy flat-array shape — still accepted for back-compat
_MOCK_MARKET_DATA_BKJY_LEGACY = {
    "ErrorId": "0",
    "Data": {
        "881386.SH": {
            "Date": ["20250603", "20250604", "20250605"],
            "BK05": ["6.50", "6.55", "6.60"],
            "BK09": ["12", "15", "10"],
        }
    },
}
# Upgraded scjy payload: per-field record list, top-level SC key
_MOCK_MARKET_DATA_SCJY = {
    "ErrorId": "0",
    "Data": {
        "SC": {
            "SC01": [
                {"Date": "20250603", "Value": ["1200000"]},
                {"Date": "20250604", "Value": ["1250000"]},
                {"Date": "20250605", "Value": ["1180000"]},
            ],
            "SC03": [
                {"Date": "20250603", "Value": ["45"]},
                {"Date": "20250604", "Value": ["38"]},
                {"Date": "20250605", "Value": ["52"]},
            ],
        }
    },
}
_MOCK_RELATION = {
    "Value": [
        {"BlockCode": "881386.SH", "BlockName": "银行", "BlockType": "行业"},
        {"BlockCode": "885001.SH", "BlockName": "沪股通", "BlockType": "概念"},
    ]
}
_MOCK_SECTOR_LIST_LT0 = ["880081.SH", "880082.SH"]
_MOCK_SECTOR_LIST_LT1 = [
    {"Code": "880081.SH", "Name": "轮动趋势"},
    {"Code": "880082.SH", "Name": "昨日强势"},
]
_MOCK_UPDOWN = {
    "UpHome": 2500,
    "DownHome": 2300,
    "Shanghai": {"UpHome": 1200, "DownHome": 1100},
    "Shenzhen": {"UpHome": 1100, "DownHome": 1000},
    "Beijing": {"UpHome": 200, "DownHome": 200},
    "Total": {"UpHome": 2500, "DownHome": 2300},
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _loader(mock_return=None):
    """Build a DataLoader with a mocked _call_tool."""
    from backtest.loaders.tdx_loader import DataLoader

    loader = DataLoader()

    def fake_call_tool(name, arguments):
        return mock_return

    loader._call_tool = fake_call_tool
    return loader


def _mock_initialize_response():
    """Build a fake successful initialize response with a session id."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Mcp-Session-Id": "test-session-123"}
    resp.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "tdx-mcp", "version": "1.0"},
        },
    }
    resp.text = json.dumps(resp.json.return_value)
    return resp


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------

class TestAvailability:
    @patch("backtest.loaders.tdx_loader.requests.post")
    def test_available_when_initialize_succeeds(self, mock_post):
        mock_post.return_value = _mock_initialize_response()

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        assert loader.is_available() is True

    @patch("backtest.loaders.tdx_loader.requests.post")
    def test_unavailable_when_all_servers_fail(self, mock_post):
        mock_post.side_effect = Exception("boom")

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        assert loader.is_available() is False


# ------------------------------------------------------------------
# OHLCV fetch — get_market_data
# ------------------------------------------------------------------

class TestFetchGetMarketData:
    def _run(self, raw=_MOCK_KDATA):
        return _loader(mock_return=raw).fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
            interval="1D",
        )

    def test_returns_dataframe_indexed_by_trade_date(self):
        result = self._run()
        assert "600000.SH" in result
        df = result["600000.SH"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index[0].date()) == "2025-06-03"

    def test_columns_are_open_high_low_close_volume(self):
        df = self._run()["600000.SH"]
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_date_range_filtered(self):
        result = _loader(mock_return=_MOCK_KDATA).fetch(
            codes=["600000.SH"],
            start_date="2025-06-04",
            end_date="2025-06-04",
            interval="1D",
        )
        df = result["600000.SH"]
        assert len(df) == 1
        assert str(df.index[0].date()) == "2025-06-04"

    def test_unknown_code_returns_empty(self):
        result = _loader(mock_return=_MOCK_KDATA).fetch(
            codes=["999999.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert "999999.SH" not in result

    def test_fallback_to_price_df_on_get_market_data_error(self):
        call_count = {}

        def fake_call_tool(name, arguments):
            call_count[name] = call_count.get(name, 0) + 1
            if name == "get_market_data":
                return None
            return _MOCK_PRICE_DF

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._call_tool = fake_call_tool
        result = loader.fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert call_count.get("get_market_data", 0) == 1
        assert call_count.get("price_df", 0) == 1
        assert "600000.SH" in result

    def test_dividend_type_passed_through(self):
        captured = {}

        def fake_call_tool(name, arguments):
            captured["name"] = name
            captured["args"] = arguments
            return _MOCK_KDATA

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._call_tool = fake_call_tool
        loader.fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
            dividend_type="front",
            fill_data=False,
        )
        assert captured["args"]["dividendType"] == "front"
        assert captured["args"]["fillData"] is False


# ------------------------------------------------------------------
# OHLCV fetch — price_df fallback
# ------------------------------------------------------------------

class TestFetchPriceDfFallback:
    def test_price_df_normalized_correctly(self):
        result = _loader(mock_return=_MOCK_PRICE_DF).fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        df = result["600000.SH"]
        assert len(df) == 3
        assert df["close"].iloc[0] == 11.17


# ------------------------------------------------------------------
# Period map — only 1d/1w/1mon/1y
# ------------------------------------------------------------------

class TestPeriodMap:
    @pytest.mark.parametrize("interval,expected", [
        ("1D", "1d"),
        ("1W", "1w"),
        ("1M", "1mon"),
        ("1Y", "1y"),
    ])
    def test_supported_intervals(self, interval, expected):
        result = _loader(mock_return=_MOCK_KDATA).fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
            interval=interval,
        )
        assert "600000.SH" in result

    def test_unknown_interval_defaults_to_1d(self):
        result = _loader(mock_return=_MOCK_KDATA).fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
            interval="5m",
        )
        assert "600000.SH" in result


# ------------------------------------------------------------------
# get_stock_list
# ------------------------------------------------------------------

class TestGetStockList:
    def test_list_type_0_returns_codes(self):
        loader = _loader(_MOCK_STOCK_LIST_LT0)
        df = loader.get_stock_list(market="50", list_type=0)
        assert list(df.columns) == ["stockCode", "stockName"]
        assert "600000.SH" in df["stockCode"].values

    def test_list_type_1_returns_code_and_name(self):
        loader = _loader(_MOCK_STOCK_LIST_LT1)
        df = loader.get_stock_list(market="50", list_type=1)
        assert len(df) == 3
        assert df.iloc[0]["stockCode"] == "600000.SH"
        assert df.iloc[0]["stockName"] == "浦发银行"

    def test_empty_response_returns_empty_df(self):
        loader = _loader([])
        df = loader.get_stock_list()
        assert df.empty
        assert list(df.columns) == ["stockCode", "stockName"]

    def test_data_wrapped_response(self):
        loader = _loader({"Data": _MOCK_STOCK_LIST_LT1})
        df = loader.get_stock_list(market="50", list_type=1)
        assert len(df) == 3
        assert df.iloc[0]["stockCode"] == "600000.SH"


# ------------------------------------------------------------------
# get_stock_list_in_sector
# ------------------------------------------------------------------

class TestGetStockListInSector:
    def test_list_type_0_returns_codes_only(self):
        loader = _loader(_MOCK_STOCK_LIST_IN_SECTOR_LT0)
        df = loader.get_stock_list_in_sector("881386.SH", list_type=0)
        assert "600000.SH" in df["stockCode"].values
        assert "平安银行" not in df["stockName"].values

    def test_list_type_1_handles_dict_format(self):
        loader = _loader(_MOCK_STOCK_LIST_IN_SECTOR_LT1)
        df = loader.get_stock_list_in_sector("881386.SH", list_type=1)
        assert len(df) == 2
        assert df.iloc[0]["stockCode"] == "600000.SH"
        assert df.iloc[0]["stockName"] == "浦发银行"

    def test_empty_response_returns_empty_df(self):
        loader = _loader([])
        df = loader.get_stock_list_in_sector("881386.SH")
        assert df.empty


# ------------------------------------------------------------------
# get_sector_list
# ------------------------------------------------------------------

class TestGetSectorList:
    def test_list_type_1_has_names_directly(self):
        loader = _loader(_MOCK_SECTOR_LIST_LT1)
        df = loader.get_sector_list(list_type=1)
        assert len(df) == 2
        assert df.iloc[0]["blockCode"] == "880081.SH"
        assert df.iloc[0]["blockName"] == "轮动趋势"

    def test_list_type_0_returns_empty_when_no_stock_info(self):
        loader = _loader(_MOCK_SECTOR_LIST_LT0)
        loader.get_stock_info = MagicMock(return_value=None)
        df = loader.get_sector_list(list_type=0)
        assert df.empty

    def test_list_type_0_resolves_names_via_get_stock_info(self):
        loader = _loader(_MOCK_SECTOR_LIST_LT0)
        loader.get_stock_info = MagicMock(return_value={"Name": "Resolved"})
        df = loader.get_sector_list(list_type=0)
        assert len(df) == 2
        assert df.iloc[0]["blockName"] == "Resolved"


# ------------------------------------------------------------------
# get_stock_sectors
# ------------------------------------------------------------------

class TestGetStockSectors:
    def test_returns_tuple_list(self):
        loader = _loader(_MOCK_RELATION)
        result = loader.get_stock_sectors("600000.SH")
        assert len(result) == 2
        assert result[0] == ("881386.SH", "银行", "行业")
        assert result[1] == ("885001.SH", "沪股通", "概念")


# ------------------------------------------------------------------
# get_stock_info
# ------------------------------------------------------------------

class TestGetStockInfo:
    def test_returns_info_dict(self):
        loader = _loader(_MOCK_STOCK_INFO)
        info = loader.get_stock_info("600000.SH")
        assert info["Name"] == "浦发银行"
        assert info["HSStockKind"] == "1"

    def test_error_response_returns_none(self):
        loader = _loader({"ErrorId": "1", "Message": "error"})
        info = loader.get_stock_info("600000.SH")
        assert info is None

    def test_df_variant(self):
        loader = _loader(_MOCK_STOCK_INFO)
        df = loader.get_stock_info_df("600000.SH")
        assert df is not None
        assert "Name" in df.columns


# ------------------------------------------------------------------
# get_market_snapshot
# ------------------------------------------------------------------

class TestGetMarketSnapshot:
    def test_returns_snapshot_dict(self):
        loader = _loader(_MOCK_SNAPSHOT)
        snap = loader.get_market_snapshot("600000.SH")
        assert snap["Now"] == "11.45"
        assert snap["Open"] == "11.20"

    def test_error_response_returns_none(self):
        loader = _loader({"ErrorId": "1"})
        assert loader.get_market_snapshot("600000.SH") is None

    def test_df_variant(self):
        loader = _loader(_MOCK_SNAPSHOT)
        df = loader.get_market_snapshot_df("600000.SH")
        assert df is not None
        assert "Now" in df.columns


# ------------------------------------------------------------------
# get_trading_calendar
# ------------------------------------------------------------------

class TestGetTradingCalendar:
    def test_returns_trade_date_column(self):
        loader = _loader(_MOCK_TRADING_CALENDAR)
        df = loader.get_trading_calendar("SH", "2025-06-03", "2025-06-05")
        assert "trade_date" in df.columns
        assert len(df) == 3

    def test_accepts_yyyymmdd_format(self):
        loader = _loader(_MOCK_TRADING_CALENDAR)
        df = loader.get_trading_calendar("SH", "20250603", "20250605")
        assert len(df) == 3

    def test_non_list_returns_empty(self):
        loader = _loader("not a list")
        df = loader.get_trading_calendar()
        assert df.empty

    def test_get_trading_dates_alias(self):
        loader = _loader(_MOCK_TRADING_CALENDAR)
        df = loader.get_trading_dates("SZ", "2025-06-03", "2025-06-05")
        assert "trade_date" in df.columns
        assert len(df) == 3


# ------------------------------------------------------------------
# get_divid_factors
# ------------------------------------------------------------------

class TestGetDividFactors:
    def test_returns_indexed_dataframe(self):
        loader = _loader(_MOCK_DIVID_FACTORS)
        df = loader.get_divid_factors("600000.SH")
        assert "trade_date" in df.columns or df.index.name == "trade_date"
        assert df.iloc[0]["Type"] == "1"
        assert df.iloc[0]["Bonus"] == "0.37"


# ------------------------------------------------------------------
# get_bkjy_value
# ------------------------------------------------------------------

class TestGetBkjyValue:
    def test_returns_dataframe_with_block_columns(self):
        loader = _loader(_MOCK_MARKET_DATA_BKJY)
        df = loader.get_bkjy_value(
            block_codes=["881386.SH"],
            field_list=["BK5", "BK9"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert "trade_date" in df.columns or df.index.name == "trade_date"
        assert any("BK5" in c for c in df.columns)

    def test_legacy_flat_array_shape_accepted(self):
        loader = _loader(_MOCK_MARKET_DATA_BKJY_LEGACY)
        df = loader.get_bkjy_value(
            block_codes=["881386.SH"],
            field_list=["BK05", "BK09"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert not df.empty
        assert any("BK05" in c for c in df.columns)

    def test_error_response_returns_empty_df(self):
        loader = _loader({"ErrorId": "1"})
        df = loader.get_bkjy_value(["881386.SH"], ["BK5"])
        assert df.empty

    def test_by_date_variant(self):
        loader = _loader(_MOCK_MARKET_DATA_BKJY)
        df = loader.get_bkjy_value_by_date(
            block_codes=["881386.SH"],
            field_list=["BK5"],
            year=0,
            mmdd=0,
        )
        assert "trade_date" in df.columns or df.index.name == "trade_date"


# ------------------------------------------------------------------
# get_scjy_value
# ------------------------------------------------------------------

class TestGetScjyValue:
    def test_returns_dataframe_with_field_columns(self):
        loader = _loader(_MOCK_MARKET_DATA_SCJY)
        df = loader.get_scjy_value(
            field_list=["SC01", "SC03"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert "trade_date" in df.columns or df.index.name == "trade_date"
        assert any("SC01" in c for c in df.columns)

    def test_by_date_variant(self):
        loader = _loader(_MOCK_MARKET_DATA_SCJY)
        df = loader.get_scjy_value_by_date(
            field_list=["SC01"], year=0, mmdd=0
        )
        assert "trade_date" in df.columns or df.index.name == "trade_date"


# ------------------------------------------------------------------
# get_market_updown_count
# ------------------------------------------------------------------

class TestGetMarketUpdownCount:
    def test_returns_updown_dict(self):
        loader = _loader(_MOCK_UPDOWN)
        result = loader.get_market_updown_count()
        assert result["UpHome"] == 2500
        assert result["DownHome"] == 2300


# ------------------------------------------------------------------
# get_more_info
# ------------------------------------------------------------------

class TestGetMoreInfo:
    def test_returns_metrics_dict(self):
        loader = _loader({"ErrorId": "0", "ZAF": "0.02", "PB_MRQ": "0.85"})
        result = loader.get_more_info("600000.SH")
        assert result["ZAF"] == "0.02"
        assert result["PB_MRQ"] == "0.85"

    def test_value_wrapped_payload(self):
        loader = _loader({
            "ErrorId": "0",
            "Value": {"ZAF": "0.02", "PB_MRQ": "0.85"},
        })
        result = loader.get_more_info("600000.SH")
        assert result["ZAF"] == "0.02"
        assert result["PB_MRQ"] == "0.85"

    def test_error_response_returns_none(self):
        loader = _loader({"ErrorId": "1"})
        assert loader.get_more_info("600000.SH") is None


# ------------------------------------------------------------------
# New research methods
# ------------------------------------------------------------------

class TestGetFinancialData:
    def test_returns_dataframe(self):
        loader = _loader(_MOCK_MARKET_DATA_PER_STOCK)
        df = loader.get_financial_data(
            stock_codes=["600000.SH"],
            field_list=["BK5"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert not df.empty
        assert any("BK5" in c for c in df.columns)

    def test_by_date_variant(self):
        loader = _loader(_MOCK_MARKET_DATA_PER_STOCK)
        df = loader.get_financial_data_by_date(
            stock_codes=["600000.SH"],
            field_list=["BK5"],
            year=0,
            mmdd=0,
        )
        assert df is not None


class TestGetGpjyValue:
    def test_returns_dataframe(self):
        loader = _loader(_MOCK_MARKET_DATA_PER_STOCK)
        df = loader.get_gpjy_value(
            stock_codes=["600000.SH"],
            field_list=["BK5"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert not df.empty

    def test_by_date_variant(self):
        loader = _loader(_MOCK_MARKET_DATA_PER_STOCK)
        df = loader.get_gpjy_value_by_date(
            stock_codes=["600000.SH"],
            field_list=["BK5"],
            year=0,
            mmdd=0,
        )
        assert df is not None


class TestGetGpOneData:
    def test_returns_dataframe(self):
        payload = {
            "ErrorId": "0",
            "Data": {
                "600000.SH": {
                    "GO1": [
                        {"Date": "20250603", "Value": ["1.20"]},
                    ],
                    "GO2": [
                        {"Date": "20250603", "Value": ["500000"]},
                    ],
                }
            },
        }
        loader = _loader(payload)
        df = loader.get_gp_one_data(
            stock_codes=["600000.SH"], field_list=["GO1", "GO2"]
        )
        assert not df.empty


class TestGetCbInfo:
    def test_returns_dataframe(self):
        payload = {
            "ErrorId": "0",
            "Data": [
                {"BondCode": "113050.SH", "StockCode": "300496.SZ", "Name": "..."}
            ],
        }
        loader = _loader(payload)
        df = loader.get_cb_info("113050.SH")
        assert df is not None
        assert not df.empty


class TestGetIpoInfo:
    def test_returns_dataframe(self):
        payload = {
            "ErrorId": "0",
            "Data": [
                {"Code": "600000.SH", "Price": "10.00", "Date": "20250603"}
            ],
        }
        loader = _loader(payload)
        df = loader.get_ipo_info(ipo_type=0, ipo_date=0)
        assert df is not None
        assert not df.empty


class TestGetGbInfo:
    def test_returns_dataframe(self):
        payload = {
            "ErrorId": "0",
            "Data": [
                {
                    "Date": "20250603",
                    "TotalShare": "2935200000",
                    "ActiveShare": "2935200000",
                }
            ],
        }
        loader = _loader(payload)
        df = loader.get_gb_info("600000.SH", count=1)
        assert df is not None
        assert not df.empty


# ------------------------------------------------------------------
# Cache / refresh
# ------------------------------------------------------------------

class TestRefresh:
    def test_refresh_cache_returns_true_on_payload(self):
        loader = _loader({"ErrorId": "0", "Data": "ok"})
        assert loader.refresh_cache() is True

    def test_refresh_cache_returns_false_on_none(self):
        loader = _loader(None)
        assert loader.refresh_cache() is False

    def test_refresh_kline_returns_true_on_payload(self):
        loader = _loader({"ErrorId": "0"})
        assert loader.refresh_kline(["600000.SH"], "1d") is True


# ------------------------------------------------------------------
# Client-action methods
# ------------------------------------------------------------------

class TestClientActions:
    def test_send_user_block(self):
        captured = {}

        def fake_call_tool(name, arguments):
            captured["name"] = name
            captured["args"] = arguments
            return {"ErrorId": "0"}

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._call_tool = fake_call_tool
        loader.send_user_block(["600000.SH"], block_code="X", show=True)
        assert captured["name"] == "send_user_block"
        assert captured["args"]["stocks"] == ["600000.SH"]
        assert captured["args"]["show"] is True

    def test_send_message(self):
        captured = {}

        def fake_call_tool(name, arguments):
            captured["args"] = arguments
            return {"ErrorId": "0"}

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._call_tool = fake_call_tool
        loader.send_message("hello")
        assert captured["args"]["message"] == "hello"

    @pytest.mark.parametrize("method,expected_tool,extra_args", [
        ("create_sector",  "create_sector",  ("name",)),
        ("delete_sector",  "delete_sector",  ()),
        ("rename_sector",  "rename_sector",  ("name",)),
        ("clear_sector",   "clear_sector",   ()),
    ])
    def test_sector_mutations(self, method, expected_tool, extra_args):
        captured = {}

        def fake_call_tool(name, arguments):
            captured["name"] = name
            captured["args"] = arguments
            return {"ErrorId": "0"}

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._call_tool = fake_call_tool
        getattr(loader, method)("TESTBK", *extra_args)
        assert captured["name"] == expected_tool
        assert captured["args"]["blockCode"] == "TESTBK"


# ------------------------------------------------------------------
# Transport: JSON-RPC 2.0 envelope parsing
# ------------------------------------------------------------------

class TestUnwrapResult:
    def test_unwraps_content_text_json(self):
        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        result = {
            "content": [
                {"type": "text", "text": json.dumps({"ErrorId": "0", "Data": [1, 2, 3]})}
            ]
        }
        assert loader._unwrap_result(result) == {"ErrorId": "0", "Data": [1, 2, 3]}

    def test_passes_through_direct_dict(self):
        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        result = {"ErrorId": "0", "Data": {"k": "v"}}
        assert loader._unwrap_result(result) == result

    def test_passes_through_bare_list(self):
        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        result = [1, 2, 3]
        assert loader._unwrap_result(result) == [1, 2, 3]


class TestTransportPost:
    @patch("backtest.loaders.tdx_loader.requests.post")
    def test_session_id_captured(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Mcp-Session-Id": "abc-123"}
        resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_post.return_value = resp

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._host = "localhost"
        loader._port = 3100
        loader._post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert loader._session_id == "abc-123"

    @patch("backtest.loaders.tdx_loader.requests.post")
    def test_session_id_sent_in_header(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
        mock_post.return_value = resp

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._host = "localhost"
        loader._port = 3100
        loader._session_id = "existing-session"
        loader._post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Mcp-Session-Id"] == "existing-session"

    @patch("backtest.loaders.tdx_loader.requests.post")
    def test_http_error_returns_none(self, mock_post):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "internal error"
        resp.headers = {}
        mock_post.return_value = resp

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        loader._host = "localhost"
        loader._port = 3100
        assert loader._post({"jsonrpc": "2.0", "id": 1, "method": "x"}) is None


# ------------------------------------------------------------------
# PascalCase / lowercase tolerance
# ------------------------------------------------------------------

class TestKeyCaseTolerance:
    def test_lowercase_keys_in_response(self):
        loader = _loader(_MOCK_STOCK_INFO_LOWER)
        info = loader.get_stock_info("600000.SH")
        assert info["name"] == "浦发银行"

    def test_lowercase_keys_in_ohlcv(self):
        raw_lowercase = {
            "ErrorId": "0",
            "Data": {
                "600000.SH": {
                    "date": ["20250603"],
                    "open":  ["11.00"],
                    "high":  ["11.19"],
                    "low":   ["10.80"],
                    "close": ["11.17"],
                    "volume":["97264624.00"],
                }
            }
        }
        result = _loader(raw_lowercase).fetch(
            codes=["600000.SH"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert "600000.SH" in result
