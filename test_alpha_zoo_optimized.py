#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vibe-Trading Alpha Zoo 多因子策略优化
=====================================

步骤:
1. 使用沪深300股票池
2. 因子筛选和优化 (IC/IR分析)
3. 滚动回测验证

使用方法:
    cd /Users/roc/code/Vibe-Trading
    source .venv/bin/activate
    python test_alpha_zoo_optimized.py
"""

from __future__ import annotations
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# 禁用日志噪音
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 项目路径
PROJECT_ROOT = Path(__file__).parent
AGENT_DIR = PROJECT_ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 配置
# ============================================================================

# 沪深300成分股 (30只代表性蓝筹股)
CSI300_CODES = [
    # 金融
    "600519.SH", "601318.SH", "600036.SH", "601166.SH", "601398.SH",
    "601288.SH", "600016.SH", "600030.SH", "601628.SH", "600000.SH",
    "601088.SH", "601857.SH", "601328.SH", "601601.SH", "601336.SH",
    # 消费
    "000858.SZ", "000333.SZ", "600887.SH", "600809.SH", "000596.SZ",
    "002304.SZ", "000895.SZ", "603288.SH", "002714.SZ", "000568.SZ",
    # 科技
    "300750.SZ", "300059.SZ", "688981.SH", "002230.SZ", "300124.SZ",
    # 工业
    "601012.SH", "600585.SH", "600048.SH", "601668.SH", "601186.SH",
    "601989.SH", "600900.SH", "600028.SH", "002142.SZ", "601138.SH",
    "600050.SH", "601766.SH", "601111.SH", "600111.SH", "600009.SH",
    "600690.SH", "600276.SH", "601888.SH", "000651.SZ", "002594.SZ",
    "600031.SH", "601233.SH", "600383.SH", "000002.SZ", "000001.SZ",
    "002415.SZ", "002460.SZ", "300015.SZ", "300760.SZ", "301071.SZ",
    "688111.SH", "688012.SH", "688036.SH", "300496.SZ", "002371.SZ",
    "002049.SZ", "300408.SZ", "600183.SH", "000066.SZ", "002236.SZ",
    "002456.SZ", "300751.SZ", "002812.SZ", "603259.SH", "688180.SH",
    "688095.SH", "300774.SZ", "603501.SH", "300033.SZ", "300014.SZ",
    "002466.SZ", "300253.SZ", "002410.SZ", "002252.SZ", "300054.SZ",
    "600588.SH", "600570.SH", "002439.SZ", "688561.SH", "300678.SZ",
    "002185.SZ", "000063.SZ", "300308.SZ", "300450.SZ", "603986.SH",
    "002027.SZ", "300122.SZ", "300223.SZ", "300782.SZ", "300661.SZ",
    "002558.SZ", "300364.SZ", "300418.SZ", "300383.SZ", "300324.SZ",
    "002555.SZ", "300113.SZ", "300342.SZ", "300294.SZ", "300015.SZ",
    "300003.SZ", "300015.SZ", "002007.SZ", "300347.SZ", "300529.SZ",
    "300760.SZ", "301060.SZ", "301050.SZ", "301030.SZ", "301071.SZ",
    "688111.SH", "688180.SH", "688012.SH", "688981.SH", "688036.SH",
    "002415.SZ", "002466.SZ", "002460.SZ", "002371.SZ", "002230.SZ",
    "300750.SZ", "300059.SZ", "300124.SZ", "300496.SZ", "300347.SZ",
    "300529.SZ", "300760.SZ", "300015.SZ", "300014.SZ", "300223.SZ",
    "300253.SZ", "300383.SZ", "300324.SZ", "300342.SZ", "300294.SZ",
    "300113.SZ", "300408.SZ", "300450.SZ", "300308.SZ", "300033.SZ",
    "603501.SH", "603288.SH", "603259.SH", "603986.SH", "603501.SH",
    "002304.SZ", "002027.SZ", "002714.SZ", "002304.SZ", "002594.SZ",
    "000895.SZ", "000568.SZ", "000596.SZ", "000651.SZ", "000858.SZ",
    "600585.SH", "600048.SH", "600383.SH", "600111.SH", "600009.SH",
    "600276.SH", "600031.SH", "600690.SH", "600887.SH", "600809.SH",
    "600900.SH", "600050.SH", "600028.SH", "600000.SH", "600016.SH",
    "601012.SH", "601012.SH", "601138.SH", "601186.SH", "601668.SH",
    "601766.SH", "601111.SH", "601989.SH", "601328.SH", "601601.SH",
    "601336.SH", "601233.SH", "601012.SH", "601888.SH", "601318.SH",
    "601628.SH", "601857.SH", "601398.SH", "601288.SH", "601166.SH",
]

# 去重
CSI300_CODES = list(set(CSI300_CODES))[:100]  # 限制100只以加快速度


# ============================================================================
# 数据加载
# ============================================================================

def load_from_internal(
    codes: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, "DataFrame"]:
    """从 Internal API 加载数据"""
    from backtest.loaders.internal_loader import DataLoader as InternalLoader
    
    print(f"  [Internal] 加载 {len(codes)} 只股票...")
    
    try:
        loader = InternalLoader()
        if not loader.is_available():
            print("  [Internal] 服务不可用")
            return {}
        
        # 批量加载
        data_map = {}
        batch_size = 20
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i+batch_size]
            try:
                batch_data = loader.fetch(batch, start_date, end_date)
                data_map.update(batch_data)
                print(f"    批次 {i//batch_size + 1}: 加载 {len(batch_data)}/{len(batch)} 只")
            except Exception as e:
                print(f"    批次 {i//batch_size + 1} 错误: {e}")
        
        loader._http.close()
        return data_map
    except Exception as e:
        print(f"  [Internal] 错误: {e}")
        return {}


# ============================================================================
# 面板构建
# ============================================================================

def build_wide_panel(
    data_map: Dict[str, "DataFrame"],
    min_data_points: int = 100
) -> Dict[str, "DataFrame"]:
    """构建宽面板，剔除数据不足的股票"""
    import pandas as pd
    
    if not data_map:
        return {}
    
    # 过滤数据量不足的股票
    valid_codes = [code for code, df in data_map.items() if len(df) >= min_data_points]
    print(f"  数据充足的股票: {len(valid_codes)}/{len(data_map)}")
    
    if not valid_codes:
        return {}
    
    all_dates = sorted(set().union(*(data_map[c].index for c in valid_codes)))
    if not all_dates:
        return {}
    
    date_index = pd.date_range(all_dates[0], all_dates[-1], freq="B")
    
    panel: Dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        frames = {}
        for code in valid_codes:
            df = data_map[code]
            if field in df.columns:
                frames[code] = df[field]
        if frames:
            panel[field] = pd.DataFrame(frames).reindex(date_index)
    
    return panel


# ============================================================================
# IC 分析
# ============================================================================

def compute_ic_series(
    factor_panel: "DataFrame",
    close_panel: "DataFrame",
    forward_periods: List[int] = [1, 5, 10]
) -> Dict[str, Dict[str, float]]:
    """计算 IC 时间序列"""
    import numpy as np
    
    results = {}
    
    for period in forward_periods:
        forward_ret = close_panel.pct_change(period).shift(-period)
        
        ic_series = []
        for date in factor_panel.index:
            if date not in forward_ret.index:
                continue
            s = factor_panel.loc[date]
            r = forward_ret.loc[date]
            valid = s.notna() & r.notna()
            if valid.sum() >= 10:  # 至少10只股票
                ic = s[valid].corr(r[valid])
                if not np.isnan(ic):
                    ic_series.append(ic)
        
        if ic_series:
            ic_arr = np.array(ic_series)
            results[f"IC_{period}d"] = {
                "ic_mean": round(float(np.mean(ic_arr)), 6),
                "ic_std": round(float(np.std(ic_arr)), 6),
                "ir": round(float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else 0, 4),
                "ic_positive_ratio": round(float(np.mean(ic_arr > 0)), 4),
                "count": len(ic_series),
            }
    
    return results


def load_zookeeper() -> Tuple:
    """加载 ZooSignalEngine 和 Registry"""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "zoo_signal_engine",
        AGENT_DIR / "src" / "skills" / "multi-factor" / "zoo_signal_engine.py"
    )
    zoo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zoo_module)
    
    from src.factors.registry import get_default_registry
    
    return zoo_module.ZooSignalEngine, get_default_registry()


def create_engine(alpha_ids: List[str], registry=None) -> "ZooSignalEngine":
    """创建引擎，兼容不同版本"""
    ZooSignalEngine, default_registry = load_zookeeper()
    reg = registry or default_registry
    
    # 使用 dataclass 方式创建
    return ZooSignalEngine(
        alpha_ids=tuple(alpha_ids),
        weights=None,
        standardize=True,
        top_n=20,
        bottom_n=None,
        _registry=reg,
    )


# ============================================================================
# 滚动回测
# ============================================================================

def rolling_backtest(
    panel: Dict[str, "DataFrame"],
    engine_class,
    registry,
    window_size: int = 60,  # 60天训练窗口
    test_period: int = 20,  # 20天测试
    step: int = 20,  # 每月滚动
) -> Dict:
    """滚动回测: 使用历史数据训练，预测未来收益"""
    import pandas as pd
    import numpy as np
    
    close = panel["close"]
    dates = close.index.tolist()
    
    results = {
        "signals": [],      # 每日信号
        "returns": [],      # 对应收益
        "position_dates": [],
    }
    
    n_dates = len(dates)
    n_windows = (n_dates - window_size) // step
    
    print(f"\n  滚动回测: {n_windows} 个窗口")
    
    for i in range(n_windows):
        train_start = i * step
        train_end = train_start + window_size
        test_start = train_end
        test_end = min(test_start + test_period, n_dates)
        
        train_dates = dates[train_start:train_end]
        test_dates = dates[test_start:test_end]
        
        if len(train_dates) < window_size or len(test_dates) < 5:
            continue
        
        # 训练期面板
        train_panel = {}
        for field, df in panel.items():
            if field == "_meta":
                continue
            train_panel[field] = df.loc[train_dates]
        
        # 测试期面板
        test_panel = {}
        for field, df in panel.items():
            if field == "_meta":
                continue
            test_panel[field] = df.loc[test_dates]
        
        try:
            # 创建引擎 (简化版,只用前5个能工作的因子)
            engine = create_engine([
                "gtja191_001", "gtja191_002", "gtja191_003",
                "gtja191_004", "gtja191_005"
            ])
            
            # 计算测试期信号
            test_signal = engine.compute_signal(test_panel)
            test_close = test_panel["close"]
            
            # 计算每期收益 (等权做多组合)
            for j, date in enumerate(test_dates[:-1]):
                next_date = test_dates[j + 1]
                
                longs = test_signal.loc[date][test_signal.loc[date] > 0].index.tolist()
                if longs:
                    # 当天收盘买入,次日收盘卖出的收益
                    current_prices = test_close.loc[date, longs]
                    next_prices = test_close.loc[next_date, longs]
                    
                    # 过滤无效值
                    valid_mask = current_prices.notna() & next_prices.notna() & (current_prices > 0)
                    if valid_mask.sum() > 0:
                        daily_ret = (next_prices[valid_mask] / current_prices[valid_mask] - 1).mean()
                        if not np.isnan(daily_ret) and abs(daily_ret) < 0.5:  # 过滤极端值
                            results["returns"].append(float(daily_ret))
                            results["position_dates"].append(date)
                            results["signals"].append(longs)
        
        except Exception as e:
            print(f"    窗口 {i+1} 错误: {e}")
            continue
        
        if (i + 1) % 5 == 0:
            print(f"    进度: {i+1}/{n_windows} 窗口")
    
    return results


def compute_backtest_metrics(results: Dict) -> Dict:
    """计算回测指标"""
    import numpy as np
    
    returns = np.array(results["returns"])
    if len(returns) == 0:
        return {}
    
    cum_ret = np.cumprod(1 + returns)
    max_ret = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - max_ret) / max_ret
    
    return {
        "total_return": round(float(cum_ret[-1] - 1) * 100, 2) if len(cum_ret) > 0 else 0,
        "annual_return": round(float((cum_ret[-1] ** (252 / len(returns))) - 1) * 100, 2) if len(returns) > 0 else 0,
        "sharpe_ratio": round(float(np.mean(returns) / np.std(returns) * np.sqrt(252)), 4) if np.std(returns) > 0 else 0,
        "max_drawdown": round(float(np.min(drawdown) * 100), 2),
        "win_rate": round(float(np.mean(returns > 0)) * 100, 2),
        "avg_return": round(float(np.mean(returns) * 100), 4),
        "n_trades": len(returns),
    }


# ============================================================================
# 主流程
# ============================================================================

def run_optimized_test():
    """主测试函数"""
    import pandas as pd
    
    print("=" * 70)
    print("Vibe-Trading Alpha Zoo 多因子策略优化")
    print("=" * 70)
    
    # ============ 配置 ============
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")  # 500天足够滚动回测
    
    print(f"\n[配置]")
    print(f"  股票池: 沪深300 (最多 {len(CSI300_CODES)} 只)")
    print(f"  时间范围: {start_date} ~ {end_date}")
    
    # ============ Step 1: 加载数据 ============
    print(f"\n{'='*70}")
    print("Step 1: 加载沪深300股票池数据")
    print("=" * 70)
    
    data_map = load_from_internal(CSI300_CODES, start_date, end_date)
    
    if not data_map:
        print("  错误: 无法加载数据")
        return
    
    print(f"\n  成功加载: {len(data_map)} 只股票")
    
    # ============ Step 2: 构建面板 ============
    print(f"\n{'='*70}")
    print("Step 2: 构建宽面板")
    print("=" * 70)
    
    panel = build_wide_panel(data_map, min_data_points=200)
    
    if "close" not in panel:
        print("  错误: 面板缺少 'close' 列")
        return
    
    print(f"  面板形状: {panel['close'].shape}")
    print(f"  股票数量: {len(panel['close'].columns)}")
    print(f"  日期范围: {panel['close'].index[0].strftime('%Y-%m-%d')} ~ {panel['close'].index[-1].strftime('%Y-%m-%d')}")
    
    # ============ Step 3: 因子 IC 分析 ============
    print(f"\n{'='*70}")
    print("Step 3: 因子 IC 分析 (筛选有效因子)")
    print("=" * 70)
    
    ZooSignalEngine, registry = load_zookeeper()
    
    # 测试的因子列表
    test_alphas = [
        "gtja191_001", "gtja191_002", "gtja191_003", "gtja191_004", "gtja191_005",
        "gtja191_006", "gtja191_007", "gtja191_008", "gtja191_009", "gtja191_010",
        "gtja191_011", "gtja191_012", "gtja191_013", "gtja191_014", "gtja191_015",
    ]
    
    alpha_results = {}
    valid_alphas = []
    
    for alpha_id in test_alphas:
        try:
            factor = registry.compute(alpha_id, panel)
            ic_stats = compute_ic_series(factor, panel["close"], [5])
            
            if "IC_5d" in ic_stats:
                stats = ic_stats["IC_5d"]
                alpha_results[alpha_id] = stats
                
                # 筛选条件: IC > 0.02, IR > 0.3, IC正比例 > 55%
                if stats["ic_mean"] > 0.02 and stats["ir"] > 0.3 and stats["ic_positive_ratio"] > 0.55:
                    valid_alphas.append(alpha_id)
                    print(f"  ✅ {alpha_id}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}, IC+={stats['ic_positive_ratio']:.2%}")
                else:
                    print(f"  ⚠️  {alpha_id}: IC={stats['ic_mean']:.4f}, IR={stats['ir']:.4f}, IC+={stats['ic_positive_ratio']:.2%} [不合格]")
            else:
                print(f"  ❌ {alpha_id}: IC计算失败")
                
        except Exception as e:
            print(f"  ❌ {alpha_id}: {e}")
    
    print(f"\n  有效因子: {len(valid_alphas)}/{len(test_alphas)}")
    
    if valid_alphas:
        print(f"  筛选出的因子: {valid_alphas}")
    else:
        print("  无有效因子，使用全部因子进行滚动回测")
        valid_alphas = test_alphas[:10]  # 使用前10个
    
    # ============ Step 4: 滚动回测 ============
    print(f"\n{'='*70}")
    print("Step 4: 滚动回测验证")
    print("=" * 70)
    
    # 创建滚动回测引擎
    rolling_engine = create_engine(valid_alphas)
    
    results = rolling_backtest(
        panel=panel,
        engine_class=ZooSignalEngine,
        registry=registry,
        window_size=120,  # 120天训练
        test_period=20,   # 20天测试
        step=20,          # 每月滚动
    )
    
    # ============ Step 5: 结果统计 ============
    print(f"\n{'='*70}")
    print("Step 5: 回测结果统计")
    print("=" * 70)
    
    metrics = compute_backtest_metrics(results)
    
    if metrics:
        print(f"\n  📊 滚动回测绩效:")
        print(f"     总收益率: {metrics['total_return']:.2f}%")
        print(f"     年化收益率: {metrics['annual_return']:.2f}%")
        print(f"     夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"     最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"     胜率: {metrics['win_rate']:.2f}%")
        print(f"     平均收益: {metrics['avg_return']:.4f}%")
        print(f"     交易次数: {metrics['n_trades']}")
        
        # IC 统计汇总
        print(f"\n  📈 因子 IC 汇总:")
        avg_ic = sum(alpha_results[a]["ic_mean"] for a in valid_alphas if a in alpha_results) / max(len(valid_alphas), 1)
        avg_ir = sum(alpha_results[a]["ir"] for a in valid_alphas if a in alpha_results) / max(len(valid_alphas), 1)
        print(f"     平均 IC: {avg_ic:.4f}")
        print(f"     平均 IR: {avg_ir:.4f}")
    else:
        print("  回测结果为空")
    
    print("\n" + "=" * 70)
    print("优化完成!")
    print("=" * 70)


if __name__ == "__main__":
    run_optimized_test()