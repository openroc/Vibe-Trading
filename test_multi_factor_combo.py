#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vibe-Trading 多因子组合回测
=========================

数据源: TDX MCP Server (包含 Amount 字段)
功能:
1. 沪深300股票池 (100只)
2. 多因子组合 (GTJA191 因子)
3. 滚动回测验证

使用方法:
    cd /Users/roc/code/Vibe-Trading
    source .venv/bin/activate
    python test_multi_factor_combo.py
"""

from __future__ import annotations
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 禁用日志噪音
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).parent
AGENT_DIR = PROJECT_ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 沪深300代表性股票
# ============================================================================

CSI300_CODES = [
    "600519.SH", "601318.SH", "600036.SH", "600016.SH", "600030.SH",
    "601166.SH", "601398.SH", "601288.SH", "600000.SH", "601628.SH",
    "601088.SH", "601857.SH", "600028.SH", "601012.SH", "600585.SH",
    "600048.SH", "601668.SH", "601186.SH", "600900.SH", "600050.SH",
    "601766.SH", "601111.SH", "601138.SH", "601601.SH", "601336.SH",
    "600009.SH", "600276.SH", "600690.SH", "600887.SH", "600809.SH",
    "000858.SZ", "000333.SZ", "000651.SZ", "000001.SZ", "000002.SZ",
    "002594.SZ", "002304.SZ", "002714.SZ", "002415.SZ", "002027.SZ",
]

# 减少到40只加快测试
CSI300_CODES = list(set(CSI300_CODES))[:40]


# ============================================================================
# TDX 数据加载
# ============================================================================

def load_from_tdx(
    codes: List[str],
    start_date: str,
    end_date: str,
    include_amount: bool = True
) -> Dict[str, "DataFrame"]:
    """从 TDX MCP Server 加载 A 股数据
    
    环境变量配置:
        TDX_MCP_HOSTS - 逗号分隔的 host 列表，默认 192.168.3.2,192.168.3.53
        TDX_MCP_PORTS - 逗号分隔的 port 列表，默认 3100,3101
        TDX_MCP_TIMEOUT - 超时秒数，默认 30
    """
    import os
    from backtest.loaders.tdx_loader import DataLoader as TDXLoader
    import asyncio
    
    hosts = os.getenv("TDX_MCP_HOSTS", "192.168.3.2,192.168.3.53").split(",")
    ports = [int(p) for p in os.getenv("TDX_MCP_PORTS", "3100,3101").split(",")]
    
    print(f"  [TDX] 尝试连接: {hosts}:{ports}")
    
    try:
        loader = TDXLoader()
        if not loader.is_available():
            print("  [TDX] MCP 服务不可用，请检查:")
            print(f"        - 网络连接: {hosts}")
            print(f"        - 端口: {ports}")
            print(f"        - MCP Server 是否运行")
            return {}
        
        print(f"  [TDX] 加载 {len(codes)} 只股票...")
        
        # 指定需要的字段 (包含 Amount)
        fields = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]
        
        # 批量加载
        data_map = {}
        batch_size = 20
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            try:
                batch_data = loader.fetch(
                    batch, 
                    start_date, 
                    end_date,
                    fields=fields
                )
                data_map.update(batch_data)
                print(f"    批次 {i//batch_size + 1}: 加载 {len(batch_data)}/{len(batch)} 只")
            except Exception as e:
                print(f"    批次 {i//batch_size + 1} 错误: {e}")
        
        return data_map
        
    except Exception as e:
        print(f"  [TDX] 错误: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_from_internal(
    codes: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, "DataFrame"]:
    """从 Internal API 加载数据 (备用)"""
    from backtest.loaders.internal_loader import DataLoader as InternalLoader
    
    print(f"  [Internal] 加载 {len(codes)} 只股票...")
    
    try:
        loader = InternalLoader()
        if not loader.is_available():
            return {}
        
        data_map = {}
        batch_size = 20
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            try:
                batch_data = loader.fetch(
                    batch, 
                    start_date, 
                    end_date,
                    fields=["amount"]  # 请求 amount 字段
                )
                data_map.update(batch_data)
            except Exception:
                pass
        
        loader._http.close()
        return data_map
    except Exception:
        return {}


# ============================================================================
# 面板构建
# ============================================================================

def build_wide_panel_with_amount(
    data_map: Dict[str, "DataFrame"],
    min_data_points: int = 100
) -> Dict[str, "DataFrame"]:
    """构建宽面板 (包含 amount 字段)"""
    import pandas as pd
    
    if not data_map:
        return {}
    
    valid_codes = [code for code, df in data_map.items() if len(df) >= min_data_points]
    print(f"  有效股票: {len(valid_codes)}/{len(data_map)}")
    
    if not valid_codes:
        return {}
    
    all_dates = sorted(set().union(*(data_map[c].index for c in valid_codes)))
    if not all_dates:
        return {}
    
    date_index = pd.date_range(all_dates[0], all_dates[-1], freq="B")
    
    panel: Dict[str, pd.DataFrame] = {}
    
    # 包含所有可能的字段
    fields_to_include = ["open", "high", "low", "close", "volume", "amount"]
    
    for field in fields_to_include:
        frames = {}
        for code in valid_codes:
            df = data_map[code]
            # 支持大小写
            for col in df.columns:
                if col.lower() == field:
                    frames[code] = df[col]
                    break
        if frames:
            panel[field] = pd.DataFrame(frames).reindex(date_index)
    
    return panel


# ============================================================================
# 引擎创建
# ============================================================================

def create_engine(alpha_ids: List[str], registry=None) -> "ZooSignalEngine":
    """创建多因子信号引擎"""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "zoo_signal_engine",
        AGENT_DIR / "src" / "skills" / "multi-factor" / "zoo_signal_engine.py"
    )
    zoo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zoo_module)
    
    ZooSignalEngine = zoo_module.ZooSignalEngine
    
    if registry is None:
        from src.factors.registry import get_default_registry
        registry = get_default_registry()
    
    return ZooSignalEngine(
        alpha_ids=tuple(alpha_ids),
        weights=None,
        standardize=True,
        top_n=5,  # 持仓前5名
        bottom_n=None,
        _registry=registry,
    )


# ============================================================================
# IC 分析
# ============================================================================

def compute_ic_stats(
    factor_panel: "DataFrame",
    close_panel: "DataFrame",
    forward_period: int = 5
) -> dict:
    """计算 IC 统计"""
    import numpy as np
    
    forward_ret = close_panel.pct_change(forward_period).shift(-forward_period)
    
    ic_series = []
    for date in factor_panel.index:
        if date not in forward_ret.index:
            continue
        s = factor_panel.loc[date]
        r = forward_ret.loc[date]
        valid = s.notna() & r.notna()
        if valid.sum() >= 10:
            ic = s[valid].corr(r[valid])
            if not np.isnan(ic):
                ic_series.append(ic)
    
    if not ic_series:
        return {}
    
    ic_arr = np.array(ic_series)
    return {
        "ic_mean": float(np.mean(ic_arr)),
        "ic_std": float(np.std(ic_arr)),
        "ir": float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else 0,
        "ic_positive_ratio": float(np.mean(ic_arr > 0)),
        "count": len(ic_series),
    }


# ============================================================================
# 滚动回测
# ============================================================================

def rolling_backtest(
    panel: Dict[str, "DataFrame"],
    alpha_ids: List[str],
    window_size: int = 120,
    test_period: int = 20,
    step: int = 20,
    warmup: int = 30,  # 额外预热期
) -> Dict:
    """滚动回测"""
    import numpy as np
    
    close = panel["close"]
    dates = close.index.tolist()
    n_dates = len(dates)
    n_windows = (n_dates - window_size) // step
    
    print(f"\n  滚动回测: {n_windows} 个窗口")
    
    results = {
        "returns": [],
        "position_dates": [],
        "equity_curve": [1.0],
    }
    
    equity = 1.0
    
    for i in range(n_windows):
        train_end = i * step + window_size
        # 包含预热期的测试窗口
        warmup_start = max(0, train_end - warmup)
        test_dates = dates[train_end:min(train_end + test_period, n_dates)]
        
        if len(test_dates) < 5:
            continue
        
        # 测试期面板 (包含预热期历史数据供因子计算)
        # 直接从原始 panel 中获取完整时间范围
        full_dates = dates[warmup_start:train_end + test_period]
        full_panel = {k: v.loc[full_dates].copy() for k, v in panel.items() if k != "_meta"}
        
        try:
            engine = create_engine(alpha_ids)
            # 使用完整面板 (包含预热期) 计算信号
            signal = engine.compute_signal(full_panel)
            test_close = full_panel["close"]
            
            # 检查信号是否有效
            if signal is None or signal.empty or signal.notna().sum().sum() < 10:
                continue
            
            # 计算每日收益
            for j, date in enumerate(test_dates[:-1]):
                next_date = test_dates[j + 1]
                
                longs = signal.loc[date][signal.loc[date] > 0].index.tolist()
                if longs:
                    curr_prices = test_close.loc[date, longs]
                    next_prices = test_close.loc[next_date, longs]
                    
                    valid = curr_prices.notna() & next_prices.notna() & (curr_prices > 0)
                    if valid.sum() > 0:
                        daily_ret = (next_prices[valid] / curr_prices[valid] - 1).mean()
                        if not np.isnan(daily_ret) and abs(daily_ret) < 0.2:
                            results["returns"].append(float(daily_ret))
                            results["position_dates"].append(date)
                            
                            # 更新权益曲线
                            equity *= (1 + daily_ret)
                            results["equity_curve"].append(equity)
        
        except Exception as e:
            print(f"    窗口 {i+1} 错误: {e}")
            continue
        
        if (i + 1) % 5 == 0:
            print(f"    进度: {i+1}/{n_windows}")
    
    return results


def compute_metrics(results: Dict) -> Dict:
    """计算回测指标"""
    import numpy as np
    
    returns = np.array(results["returns"])
    equity = np.array(results["equity_curve"])
    
    if len(returns) == 0:
        return {}
    
    total_ret = equity[-1] - 1
    ann_ret = (equity[-1] ** (252 / len(returns))) - 1 if len(returns) > 0 else 0
    
    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = np.min(drawdown)
    
    # 夏普
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return {
        "total_return": round(float(total_ret * 100), 2),
        "annual_return": round(float(ann_ret * 100), 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd * 100), 2),
        "win_rate": round(float(np.mean(returns > 0) * 100), 2),
        "avg_return": round(float(np.mean(returns) * 100), 4),
        "n_trades": len(returns),
    }


# ============================================================================
# 主流程
# ============================================================================

def run_multi_factor_combo_test():
    """多因子组合回测主函数"""
    print("=" * 70)
    print("Vibe-Trading 多因子组合回测")
    print("=" * 70)
    
    # 配置
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    
    print(f"\n[配置]")
    print(f"  股票池: 沪深300 ({len(CSI300_CODES)} 只)")
    print(f"  时间范围: {start_date} ~ {end_date}")
    
    # Step 1: 数据加载 - 优先使用 TDX MCP Server
    print(f"\n{'='*70}")
    print("Step 1: 数据加载")
    print("=" * 70)
    
    # 首先尝试 TDX
    data_map = load_from_tdx(CSI300_CODES[:10], start_date, end_date)  # 测试几只
    if len(data_map) >= 5:
        # TDX 可用，加载全部
        data_map = load_from_tdx(CSI300_CODES, start_date, end_date)
    else:
        # TDX 不可用，使用 Internal API
        print("  TDX 不可用，使用 Internal API...")
        data_map = load_from_internal(CSI300_CODES, start_date, end_date)
    
    if not data_map:
        print("  错误: 无法加载数据")
        return
    
    print(f"\n  成功加载: {len(data_map)} 只股票")
    
    # 检查 amount 字段
    sample_code = list(data_map.keys())[0]
    sample_df = data_map[sample_code]
    has_amount = "amount" in sample_df.columns or "Amount" in sample_df.columns
    print(f"  包含 amount 字段: {has_amount}")
    if has_amount:
        print(f"  样本列名: {list(sample_df.columns)}")
    
    # Step 2: 构建面板
    print(f"\n{'='*70}")
    print("Step 2: 构建宽面板")
    print("=" * 70)
    
    panel = build_wide_panel_with_amount(data_map)
    
    if "close" not in panel:
        print("  错误: 缺少 close 字段")
        return
    
    print(f"  面板形状: {panel['close'].shape}")
    print(f"  包含字段: {list(panel.keys())}")
    
    # Step 3: 多因子 IC 分析
    print(f"\n{'='*70}")
    print("Step 3: 多因子 IC 分析")
    print("=" * 70)
    
    # 测试因子列表 (基于批量 IC 测试的低 NaN 率优质因子)
    test_alphas = [
        "gtja191_136", "gtja191_076", "gtja191_168", 
        "gtja191_139", "gtja191_113", "gtja191_005",
    ]
    
    from src.factors.registry import get_default_registry
    registry = get_default_registry()
    
    alpha_results = {}
    valid_alphas = []
    
    for alpha_id in test_alphas:
        try:
            factor = registry.compute(alpha_id, panel)
            stats = compute_ic_stats(factor, panel["close"], 5)
            
            if stats:
                alpha_results[alpha_id] = stats
                
                # 筛选: IC > 0.01, IR > 0.05, IC+ > 50%
                if stats["ic_mean"] > 0.01 and stats["ir"] > 0.05 and stats["ic_positive_ratio"] > 0.50:
                    valid_alphas.append(alpha_id)
                    print(f"  ✅ {alpha_id}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}, IC+={stats['ic_positive_ratio']:.0%}")
                else:
                    print(f"  ⚠️  {alpha_id}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}, IC+={stats['ic_positive_ratio']:.0%}")
            else:
                print(f"  ❌ {alpha_id}: IC计算失败")
                
        except Exception as e:
            print(f"  ❌ {alpha_id}: {e}")
    
    print(f"\n  有效因子: {len(valid_alphas)}/{len(test_alphas)}")
    
    if not valid_alphas:
        print("  无有效因子，使用全部因子")
        valid_alphas = test_alphas
    
    # Step 4: 组合回测
    print(f"\n{'='*70}")
    print(f"Step 4: 组合因子回测 ({len(valid_alphas)} 个因子)")
    print("=" * 70)
    
    print(f"  使用因子: {valid_alphas}")
    
    results = rolling_backtest(
        panel=panel,
        alpha_ids=valid_alphas,
        window_size=120,
        test_period=20,
        step=20,
    )
    
    # Step 5: 结果统计
    print(f"\n{'='*70}")
    print("Step 5: 回测结果")
    print("=" * 70)
    
    metrics = compute_metrics(results)
    
    if metrics:
        print(f"\n  📊 绩效指标:")
        print(f"     总收益率:    {metrics['total_return']:>8.2f}%")
        print(f"     年化收益率: {metrics['annual_return']:>8.2f}%")
        print(f"     夏普比率:   {metrics['sharpe_ratio']:>8.4f}")
        print(f"     最大回撤:   {metrics['max_drawdown']:>8.2f}%")
        print(f"     胜率:       {metrics['win_rate']:>8.2f}%")
        print(f"     平均收益:   {metrics['avg_return']:>8.4f}%/天")
        print(f"     交易次数:   {metrics['n_trades']:>8d}")
        
        # 因子 IC 汇总
        import numpy as np
        avg_ic = np.mean([alpha_results[a]["ic_mean"] for a in valid_alphas if a in alpha_results])
        avg_ir = np.mean([alpha_results[a]["ir"] for a in valid_alphas if a in alpha_results])
        avg_ic_pos = np.mean([alpha_results[a]["ic_positive_ratio"] for a in valid_alphas if a in alpha_results])
        
        print(f"\n  📈 组合因子统计:")
        print(f"     平均 IC:      {avg_ic:>8.4f}")
        print(f"     平均 IR:      {avg_ir:>8.4f}")
        print(f"     平均 IC+率:   {avg_ic_pos:>8.2%}")
        
        # 评级
        print(f"\n  🎯 综合评级:")
        score = 0
        if metrics['annual_return'] > 10: score += 1
        if metrics['sharpe_ratio'] > 0.5: score += 1
        if abs(metrics['max_drawdown']) < 15: score += 1
        if metrics['win_rate'] > 50: score += 1
        print(f"     {'⭐' * score}{'☆' * (4-score)} ({score}/4)")
    
    print("\n" + "=" * 70)
    print("回测完成!")
    print("=" * 70)


if __name__ == "__main__":
    run_multi_factor_combo_test()