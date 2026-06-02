---
name: data-routing
category: data-source
description: Data source selection decision tree. Load this skill BEFORE any backtest or data-fetching task to choose the best available data source.
---

## Data Source Overview

| Source | Markets | Auth Required | Network | Skill |
|--------|---------|---------------|---------|-------|
| tdx | A-shares (real-time + history) | No (via MCP server) | Internal network | — |
| internal | A-shares, US, crypto | No | Internal network | — |
| tushare | A-shares, funds, futures, macro | Yes (`TUSHARE_TOKEN`) | China network | tushare |
| akshare | A-shares, US, HK, futures, macro, forex | No | Unrestricted | akshare |
| yfinance | US stocks, HK stocks, ETFs | No | Needs Yahoo Finance access | yfinance |
| okx | Crypto (OKX exchange) | No | Needs okx.com access | okx-market |
| ccxt | Crypto (100+ exchanges) | No | Needs exchange access | ccxt |

## Decision Tree

### Backtest Scenario (writing config.json)

Use `source: "auto"` — the runner automatically routes by symbol pattern and falls back to alternative sources if the primary one is unavailable.

You do NOT need to specify a concrete data source in config.json unless the user explicitly asks for one.

### Analysis / Research Scenario (writing Python scripts)

1. Identify the market type from the user's request
2. Pick the source by priority:

**A-shares**: tdx (MCP server, fastest) > internal (internal network, free) > tushare (if TUSHARE_TOKEN is set) > akshare (free fallback)

**US stocks**: yfinance > akshare

**HK stocks**: yfinance > akshare

**Crypto**: okx (single exchange) > ccxt (multi-exchange) > internal

**Futures**: tushare > akshare

**Macro / economics**: akshare > tushare

**Forex**: akshare > yfinance

3. Load the corresponding skill for API details: `load_skill("akshare")`

### Availability Check

- **tdx**: check if TDX MCP server is reachable (`TDX_MCP_HOSTS:3100/3101` accessible)
- **internal**: check if internal API is accessible
- **tushare**: check if `TUSHARE_TOKEN` environment variable exists
- **yfinance / okx / ccxt / akshare**: free but may have network restrictions
- If the user reports "connection timeout" or "cannot access", switch to the same-market fallback

## Symbol Format Reference

| Market | Format | Examples |
|--------|--------|---------|
| A-shares | `NNNNNN.SZ/SH/BJ` | 000001.SZ, 600000.SH |
| US stocks | `TICKER.US` | AAPL.US, MSFT.US |
| HK stocks | `NNN(N).HK` | 700.HK, 9988.HK |
| Crypto | `SYMBOL-USDT` | BTC-USDT, ETH-USDT |
| Futures | `XXNNNN.EXCHANGE` | CU2406.SHFE |
| Forex | `XXX/YYY` | USD/CNY, EUR/USD |

## Fallback Chain (Runner Layer)

The backtest runner implements automatic fallback at the market level:

```
User requests 000001.SZ (A-share)
  -> detect market: a_share
  -> try tdx: MCP server reachable -> use tdx
  -> success

User requests 600000.SH (A-share)
  -> detect market: a_share
  -> try tdx: MCP server unreachable -> skip
  -> try internal: internal API reachable -> use internal
  -> success

User requests AAPL.US (US stock)
  -> detect market: us_equity
  -> try yfinance: available -> use yfinance
  -> success (zero config required)
```

This is transparent to the user — they just see results.

## TDX MCP Configuration

The TDX loader targets the upgraded MCP server (protocol `2024-11-05`)
exposed over the **streamable-HTTP** transport — `POST /mcp` with
JSON-RPC 2.0 bodies and a session id carried in the `Mcp-Session-Id`
header. The legacy SSE endpoint (`/sse`) is no longer available.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TDX_MCP_HOSTS` | `192.168.3.2,192.168.3.53` | Comma-separated MCP server IPs (failover order) |
| `TDX_MCP_PORTS` | `3100,3101` | Comma-separated ports to try per host |
| `TDX_MCP_TIMEOUT` | `60` | Per-request timeout in seconds |

The loader tries every `host × port` combination during the
`initialize` handshake and pins itself to the first server that
responds. A later call failure triggers another full sweep.

### Tools Exposed by the Upgraded Server

The loader wraps 30+ TDX MCP tools; the most-used are listed below.
Refer to `agent/backtest/loaders/tdx_loader.py` for full signatures
and field dictionaries.

| Category | Tools |
|----------|-------|
| **OHLCV** (runner-facing) | `get_market_data`, `price_df` (legacy fallback) |
| **Universe** | `get_stock_list`, `get_stock_list_in_sector`, `get_sector_list`, `get_relation` |
| **Reference data** | `get_stock_info`, `get_market_snapshot`, `get_trading_calendar`, `get_trading_dates`, `get_divid_factors` |
| **Financials** | `get_financial_data`, `get_financial_data_by_date` |
| **Operating metrics** | `get_gpjy_value`, `get_gpjy_value_by_date`, `get_gp_one_data` |
| **Sector metrics** | `get_bkjy_value`, `get_bkjy_value_by_date` |
| **Market metrics** | `get_scjy_value`, `get_scjy_value_by_date` |
| **Market breadth** | `get_market_updown_count`, `get_more_info` |
| **Convertible bonds / IPO / share capital** | `get_cb_info`, `get_ipo_info`, `get_gb_info` |
| **Cache control** | `refresh_cache`, `refresh_kline`, `download_financial_data` |
| **Client actions** (require a connected TDX client) | `send_user_block`, `send_message`, `create_sector`, `delete_sector`, `rename_sector`, `clear_sector` |

### `fetch()` Parameters

The backtest runner calls `loader.fetch(codes, start_date, end_date, ...)`.
Available keyword arguments (added/clarified by the upgrade):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | str | `"1D"` | TDX only supports `1D` (→`1d`), `1W` (→`1w`), `1M` (→`1mon`), `1Y` (→`1y`). Intraday bars are not available. |
| `fields` | list[str] | `None` | Optional field filter. Recognised: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Amount`, `Time`, `VolInStock`, `ForwardFactor`. |
| `dividend_type` | str | `"none"` | Forward-adjustment mode: `"none"` (raw), `"front"`, `"back"`. |
| `fill_data` | bool | `True` | Backfill missing bars with the last known close. |

### Response Shape

Two TDX payload shapes coexist on the upgraded server; the loader
normalises both transparently via `_unwrap_result()` / `_parse_value_payload()`.

* **OHLCV** (`get_market_data` / `price_df`):
  `{"ErrorId": "0", "Data": {"<code>": {"ErrorId": "0", "Date": [...], "Open": [...], ...}}}`.
  *New fields* the upgraded server may include: `Time`, `VolInStock`, `ForwardFactor`.
* **Value tools** (`get_bkjy_value`, `get_scjy_value`, `get_gpjy_value`, `get_financial_data`):
  `{"ErrorId": "0", "Data": {"<code>": {"<field>": [{"Date": "...", "Value": [scalar]}, ...]}}}`.
  Field names drop the leading zero on the upgraded server (e.g. `BK05` → `BK5`).
  The legacy flat-array shape (`{"Date": [...], "BK05": [...]}`) is still
  accepted for back-compat.

### Direct API Example

```python
from backtest.loaders.tdx_loader import DataLoader

loader = DataLoader()
if not loader.is_available():
    raise RuntimeError("No TDX MCP server reachable")

# OHLCV (used by the backtest runner)
ohlcv = loader.fetch(
    codes=["600000.SH"],
    start_date="2025-06-01",
    end_date="2025-06-30",
    interval="1D",
    dividend_type="front",
    fill_data=True,
)

# Sector metrics
df = loader.get_bkjy_value(
    block_codes=["881386.SH"],
    field_list=["BK5", "BK9"],
    start_date="20250601",
    end_date="20250630",
)

# Real-time snapshot
snap = loader.get_market_snapshot("600000.SH")
```
