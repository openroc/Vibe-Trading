"""Tests for TDX MCP loader — all external calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ------------------------------------------------------------------
# TDX response fixtures (realistic shapes from /tools schema)
# ------------------------------------------------------------------

_MOCK_KDATA = {
    "ErrorId": "0",
    "Data": {
        "600000.SH": {
            "Date": ["20250603", "20250604", "20250605"],
            "Open":  ["11.00", "11.20", "11.30"],
            "High":  ["11.19", "11.50", "11.60"],
            "Low":   ["10.80", "11.10", "11.20"],
            "Close": ["11.17", "11.45", "11.55"],
            "Volume":["97264624.00", "85000000.00", "91000000.00"],
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
    "Min": "11.10",
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
_MOCK_MARKET_DATA_BKJY = {
    "ErrorId": "0",
    "Data": {
        "881386.SH": {
            "Date": ["20250603", "20250604", "20250605"],
            "BK05": ["6.50", "6.55", "6.60"],
            "BK09": ["12", "15", "10"],
        }
    }
}
_MOCK_MARKET_DATA_SCJY = {
    "ErrorId": "0",
    "Data": {
        "SC": {
            "Date": ["20250603", "20250604", "20250605"],
            "SC01": ["1200000", "1250000", "1180000"],
            "SC03": ["45", "38", "52"],
        }
    }
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
_MOCK_UPDOWN = {"UpHome": 2500, "DownHome": 2300}


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


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------

class TestAvailability:
    @patch("httpx.Client")
    def test_available_when_sse_responds(self, mock_httpx):
        mock_httpx.return_value.__enter__.return_value.get.return_value.status_code = 200

        from backtest.loaders.tdx_loader import DataLoader

        loader = DataLoader()
        assert loader.is_available() is True

    @patch("httpx.Client")
    def test_unavailable_when_all_servers_fail(self, mock_httpx):
        mock_httpx.return_value.__enter__.return_value.get.side_effect = Exception("boom")

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
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

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
            field_list=["BK05", "BK09"],
            start_date="2025-06-03",
            end_date="2025-06-05",
        )
        assert "trade_date" in df.columns or "trade_date" == df.index.name
        assert any("BK05" in c for c in df.columns)

    def test_error_response_returns_empty_df(self):
        loader = _loader({"ErrorId": "1"})
        df = loader.get_bkjy_value(["881386.SH"], ["BK05"])
        assert df.empty


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
        assert "trade_date" in df.columns or "trade_date" == df.index.name
        assert any("SC01" in c for c in df.columns)


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

    def test_error_response_returns_none(self):
        loader = _loader({"ErrorId": "1"})
        assert loader.get_more_info("600000.SH") is None


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