#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vibe-Trading Alpha Zoo 多因子信号测试 (真实数据版)

数据源: TDX MCP Server + Internal API
测试组合多个 GTJA191 因子生成复合信号

使用方法:
    cd /Users/roc/code/Vibe-Trading/agent
    source .venv/bin/activate
    python ../test_alpha_zoo.py

数据源配置:
    - TDX:      TDX_MCP_HOSTS=192.168.3.2,192.168.3.53
    - Internal: INTERNAL_API_HOST=192.168.3.80
"""

from __future__ import annotations
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

# 禁用 mcp 库的日志噪音
logging.getLogger("mcp").setLevel(logging.WARNING)

# 项目路径
PROJECT_ROOT = Path(__file__).parent
AGENT_DIR = PROJECT_ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 数据加载
# ============================================================================

def load_from_internal(
    codes: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, "DataFrame"]:
    """从 Internal API 加载 A 股数据 (前复权 OHLCV)"""
    from backtest.loaders.internal_loader import DataLoader as InternalLoader
    from backtest.loaders.base import NoAvailableSourceError
    
    print(f"  [Internal] 尝试连接 {InternalLoader.__module__}...")
    
    try:
        loader = InternalLoader()
        if not loader.is_available():
            print("  [Internal] 服务不可用")
            return {}
        
        print("  [Internal] 连接成功，加载数据...")
        data_map = loader.fetch(codes, start_date, end_date)
        loader._http.close()
        return data_map
    except Exception as e:
        print(f"  [Internal] 错误: {e}")
        return {}


def load_from_tdx(
    codes: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, "DataFrame"]:
    """从 TDX MCP Server 加载 A 股数据"""
    from backtest.loaders.tdx_loader import DataLoader as TDXLoader
    import asyncio
    
    print(f"  [TDX] 尝试连接 TDX MCP...")
    
    try:
        loader = TDXLoader()
        if not loader.is_available():
            print("  [TDX] MCP 服务不可用")
            return {}
        
        print("  [TDX] 连接成功，加载数据...")
        data_map = loader.fetch(codes, start_date, end_date)
        
        # 清理连接
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(loader._cleanup_async())
        
        return data_map
    except Exception as e:
        print(f"  [TDX] 错误: {e}")
        return {}


def generate_demo_data(
    codes: List[str],
    start_date: str,
    end_date: str,
    seed: int = 42
) -> Dict[str, "DataFrame"]:
    """生成演示用的 OHLCV 数据"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start_date, end_date, freq="B")
    rng = np.random.default_rng(seed)
    data_map = {}
    
    for code in codes:
        base_price = 50 + rng.integers(0, 100)
        returns = rng.standard_normal(len(dates)) * 0.02
        close_prices = base_price * np.exp(np.cumsum(returns))
        trend = np.linspace(0, 0.15, len(dates))
        close_prices *= (1 + trend)
        
        df = pd.DataFrame(
            {
                "open": close_prices * (1 + rng.uniform(-0.01, 0.01, len(dates))),
                "high": close_prices * (1 + rng.uniform(0, 0.02, len(dates))),
                "low": close_prices * (1 - rng.uniform(0, 0.02, len(dates))),
                "close": close_prices,
                "volume": rng.integers(1_000_000, 10_000_000, len(dates)),
            },
            index=dates,
        )
        data_map[code] = df
    
    return data_map


# ============================================================================
# 面板构建
# ============================================================================

def build_wide_panel(
    data_map: Dict[str, "DataFrame"]
) -> Dict[str, "DataFrame"]:
    """将 data_map 转换为宽面板格式"""
    import pandas as pd
    
    if not data_map:
        return {}
    
    codes = list(data_map.keys())
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    date_index = pd.date_range(all_dates[0], all_dates[-1], freq="B")
    
    panel: Dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        frames = {}
        for code in codes:
            df = data_map[code]
            if field in df.columns:
                frames[code] = df[field]
        if frames:
            panel[field] = pd.DataFrame(frames).reindex(date_index)
    
    return panel


# ============================================================================
# IC 统计
# ============================================================================

def compute_ic_stats(
    signal_panel: "DataFrame",
    close_panel: "DataFrame",
    forward_period: int = 5
) -> dict:
    """计算 IC 统计"""
    import numpy as np
    
    forward_ret = close_panel.pct_change(forward_period).shift(-forward_period)
    
    ic_series = []
    for date in signal_panel.index:
        if date not in forward_ret.index:
            continue
        s = signal_panel.loc[date]
        r = forward_ret.loc[date]
        valid = s.notna() & r.notna()
        if valid.sum() >= 3:
            ic = s[valid].corr(r[valid])
            if not np.isnan(ic):
                ic_series.append(ic)
    
    if not ic_series:
        return {"ic_mean": 0, "ic_std": 0, "ir": 0, "ic_positive_ratio": 0, "n_samples": 0}
    
    ic_arr = np.array(ic_series)
    return {
        "ic_mean": round(float(np.mean(ic_arr)), 6),
        "ic_std": round(float(np.std(ic_arr)), 6),
        "ir": round(float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else 0, 4),
        "ic_positive_ratio": round(float(np.mean(ic_arr > 0)), 4),
        "n_samples": len(ic_series),
    }


# ============================================================================
# 主测试
# ============================================================================

def run_test():
    """主测试函数"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("=" * 60)
    print("Vibe-Trading Alpha Zoo 多因子信号测试 (真实数据)")
    print("=" * 60)
    
    # ============ 配置 ============
    test_codes = [
        "600519.SH",  # 贵州茅台
        "601318.SH",  # 中国平安
        "600036.SH",  # 招商银行
        "000333.SZ",  # 美的集团
        "000858.SZ",  # 五粮液
    ]
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # GTJA191 因子列表
    alpha_ids = [
        "gtja191_001",
        "gtja191_002",
        "gtja191_003",
        "gtja191_004",
        "gtja191_005",
    ]
    
    print(f"\n[配置]")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  股票列表: {test_codes}")
    print(f"  Alpha列表: {alpha_ids}")
    
    # ============ 数据加载 ============
    print(f"\n[1] 数据加载...")
    data_map = {}
    data_source = "未知"
    
    # 优先级: Internal > TDX > Demo
    data_map = load_from_internal(test_codes, start_date, end_date)
    if data_map:
        data_source = "Internal API"
    
    if not data_map:
        data_map = load_from_tdx(test_codes, start_date, end_date)
        if data_map:
            data_source = "TDX MCP"
    
    if not data_map:
        print("  [Demo] 使用演示数据...")
        data_map = generate_demo_data(test_codes, start_date, end_date)
        data_source = "演示数据"
    
    print(f"  数据源: {data_source}")
    print(f"  加载股票数: {len(data_map)}")
    for code, df in data_map.items():
        print(f"    {code}: {len(df)} 条数据 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    
    # ============ 构建宽面板 ============
    print(f"\n[2] 构建宽面板...")
    panel = build_wide_panel(data_map)
    
    if "close" not in panel:
        print("  错误: 面板缺少 'close' 列")
        return
    
    print(f"  面板形状: {panel['close'].shape}")
    print(f"  日期范围: {panel['close'].index[0].strftime('%Y-%m-%d')} ~ {panel['close'].index[-1].strftime('%Y-%m-%d')}")
    
    # ============ ZooSignalEngine ============
    print(f"\n[3] 创建 ZooSignalEngine...")
    
    try:
        # 动态加载 zoo_signal_engine
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "zoo_signal_engine",
            AGENT_DIR / "src" / "skills" / "multi-factor" / "zoo_signal_engine.py"
        )
        zoo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zoo_module)
        ZooSignalEngine = zoo_module.ZooSignalEngine
        
        # 创建引擎
        engine = ZooSignalEngine.from_zoo(
            alpha_ids=alpha_ids,
            weights=None,          # 等权
            standardize=True,      # 截面 z-score 标准化
            top_n=2,              # 做多 Top-2
            bottom_n=None,        # 不做空
        )
        
        print(f"  引擎配置:")
        print(f"    alpha_ids: {list(engine.alpha_ids)}")
        print(f"    standardize: {engine.standardize}")
        print(f"    top_n: {engine.top_n}")
        
    except Exception as e:
        print(f"  错误: 无法加载 ZooSignalEngine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 计算信号 ============
    print(f"\n[4] 计算复合信号...")
    
    try:
        positions = engine.compute_signal(panel)
        print(f"  输出形状: {positions.shape}")
        
        long_count = int((positions == 1.0).sum().sum())
        neutral_count = int((positions == 0.0).sum().sum())
        
        print(f"  做多信号: {long_count}")
        print(f"  中性信号: {neutral_count}")
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 展示信号 ============
    print(f"\n[5] 最近 5 天信号:")
    for date in positions.index[-5:]:
        longs = positions.loc[date][positions.loc[date] > 0].index.tolist()
        if longs:
            print(f"  {date.strftime('%Y-%m-%d')}: 做多 {longs}")
        else:
            print(f"  {date.strftime('%Y-%m-%d')}: 无信号")
    
    # ============ IC 统计 ============
    print(f"\n[6] IC 统计 (前向 5 日收益):")
    ic_stats = compute_ic_stats(positions, panel["close"], forward_period=5)
    print(f"  IC 均值: {ic_stats['ic_mean']:.4f}")
    print(f"  IC 标准差: {ic_stats['ic_std']:.4f}")
    print(f"  IR (信息比率): {ic_stats['ir']:.4f}")
    print(f"  IC 正比例: {ic_stats['ic_positive_ratio']:.2%}")
    print(f"  样本数: {ic_stats['n_samples']}")
    
    # ============ 回测引擎适配 ============
    print(f"\n[7] 回测引擎信号输出:")
    try:
        signal_dict = engine.generate(data_map)
        print(f"  生成信号数: {len(signal_dict)}")
        for code, signal in signal_dict.items():
            non_zero = int((signal != 0).sum())
            print(f"  {code}: {non_zero}/{len(signal)} 非零信号")
    except Exception as e:
        print(f"  适配器警告: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()