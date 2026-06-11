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

The TDX loader supports multiple servers with automatic failover:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TDX_MCP_HOSTS` | `192.168.3.2,192.168.3.53` | Comma-separated MCP server IPs |
| `TDX_MCP_PORTS` | `3100,3101` | Comma-separated ports to try per host |
| `TDX_MCP_TIMEOUT` | `30` | Connection timeout in seconds |
