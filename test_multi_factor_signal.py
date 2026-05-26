# -*- coding: utf-8 -*-
"""
Vibe-Trading Alpha Zoo 多因子信号测试

数据源: TDX (通达信) + Internal (内网)
测试组合多个 GTJA191 因子生成复合信号
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入数据加载器
from backtest.loaders.registry import resolve_loader, get_loader_cls_with_fallback

# 导入 Alpha Zoo 信号引擎
from src.skills.multi_factor.zoo_signal_engine import ZooSignalEngine
from src.factors.registry import Registry


def load_data_via_tdx(codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """通过 TDX 加载 A 股数据"""
    print(f"\n📡 通过 TDX 加载数据: {codes}")
    
    # 获取 TDX loader class
    try:
        loader_cls = get_loader_cls_with_fallback("tdx")
        loader = loader_cls()
    except Exception as e:
        print(f"⚠️ TDX 不可用: {e}")
        return {}
    
    if not loader.is_available():
        print("⚠️ TDX 服务未连接")
        return {}
    
    # 加载数据
    data_map = {}
    for code in codes:
        try:
            df = loader.fetch([code], start_date, end_date)
            if df is not None and not df.empty:
                data_map[code] = df
                print(f"  ✅ {code}: {len(df)} 条数据")
        except Exception as e:
            print(f"  ❌ {code}: {e}")
    
    return data_map


def load_data_via_internal(codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """通过 Internal 内网服务加载数据"""
    print(f"\n📡 通过 Internal 内网加载数据: {codes}")
    
    try:
        loader_cls = get_loader_cls_with_fallback("internal")
        loader = loader_cls()
    except Exception as e:
        print(f"⚠️ Internal 不可用: {e}")
        return {}
    
    if not loader.is_available():
        print("⚠️ Internal 服务未连接")
        return {}
    
    # 加载数据
    data_map = {}
    for code in codes:
        try:
            df = loader.fetch([code], start_date, end_date)
            if df is not None and not df.empty:
                data_map[code] = df
                print(f"  ✅ {code}: {len(df)} 条数据")
        except Exception as e:
            print(f"  ❌ {code}: {e}")
    
    return data_map


def build_wide_panel(data_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """将 data_map 转换为宽面板格式"""
    if not data_map:
        return {}
    
    codes = list(data_map.keys())
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    date_index = pd.DatetimeIndex(all_dates)
    
    panel: dict[str, pd.DataFrame] = {}
    fields = ["open", "high", "low", "close", "volume"]
    
    for field in fields:
        frames = {}
        for code in codes:
            df = data_map[code]
            if field in df.columns:
                frames[code] = df[field]
        if frames:
            panel[field] = pd.DataFrame(frames).reindex(date_index)
    
    return panel


def compute_ic_for_signal(
    signal_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    forward_period: int = 1
) -> dict:
    """计算信号 IC"""
    # 前向收益
    forward_ret = close_panel.pct_change(forward_period).shift(-forward_period)
    
    # 按日期计算 IC
    ic_series = []
    for date in signal_panel.index:
        if date not in forward_ret.index:
            continue
        signal_slice = signal_panel.loc[date]
        ret_slice = forward_ret.loc[date]
        
        valid = signal_slice.notna() & ret_slice.notna()
        if valid.sum() < 3:
            continue
            
        ic = signal_slice[valid].corr(ret_slice[valid])
        if not np.isnan(ic):
            ic_series.append(ic)
    
    if not ic_series:
        return {"ic_mean": 0, "ic_std": 0, "ir": 0, "ic_positive_ratio": 0}
    
    ic_arr = np.array(ic_series)
    return {
        "ic_mean": float(np.mean(ic_arr)),
        "ic_std": float(np.std(ic_arr)),
        "ir": float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else 0,
        "ic_positive_ratio": float(np.mean(ic_arr > 0))
    }


def main():
    print("=" * 60)
    print("Vibe-Trading Alpha Zoo 多因子信号测试")
    print("=" * 60)
    
    # ============ 配置参数 ============
    # 测试股票列表 (茅台、平安、招商银行等蓝筹股)
    test_codes = ["600519.SH", "601318.SH", "600036.SH", "000333.SZ", "000858.SZ"]
    
    # 时间范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # 选择的 Alpha IDs (GTJA191 因子)
    alpha_ids = [
        "gtja191_001",  # 收盘价
        "gtja191_002",  # 成交量
        "gtja191_003",  # 价格变动
        "gtja191_004",  # 动量
        "gtja191_005",  # 波动率
    ]
    
    # ============ 加载数据 ============
    print(f"\n📅 时间范围: {start_date} ~ {end_date}")
    
    # 尝试 TDX
    data_map = load_data_via_tdx(test_codes, start_date, end_date)
    
    # 如果 TDX 不可用，尝试 Internal
    if not data_map:
        data_map = load_data_via_internal(test_codes, start_date, end_date)
    
    if not data_map:
        print("\n⚠️ 所有数据源不可用，生成演示数据...")
        data_map = generate_demo_data(test_codes, start_date, end_date)
    
    # ============ 构建宽面板 ============
    print("\n🔧 构建宽面板...")
    panel = build_wide_panel(data_map)
    
    if "close" not in panel:
        print("❌ 面板缺少 close 数据")
        return
    
    print(f"  面板形状: {panel['close'].shape}")
    print(f"  股票数量: {len(data_map)}")
    print(f"  日期范围: {panel['close'].index[0]} ~ {panel['close'].index[-1]}")
    
    # ============ 构建多因子信号引擎 ============
    print("\n🎯 构建 ZooSignalEngine...")
    
    # 使用 ZooSignalEngine.from_zoo() 创建引擎
    engine = ZooSignalEngine.from_zoo(
        alpha_ids=alpha_ids,
        weights=None,  # 等权
        standardize=True,  # 截面 z-score 标准化
        top_n=2,  # 做多 Top-2
        bottom_n=None,  # 不做空
    )
    
    print(f"  因子数量: {len(engine.alpha_ids)}")
    print(f"  因子列表: {list(engine.alpha_ids)}")
    print(f"  标准化: {engine.standardize}")
    print(f"  Top-N: {engine.top_n}")
    
    # ============ 计算复合信号 ============
    print("\n📊 计算复合信号...")
    
    try:
        positions = engine.compute_signal(panel)
        print(f"  信号形状: {positions.shape}")
        
        # 统计信号分布
        long_count = (positions == 1.0).sum().sum()
        neutral_count = (positions == 0.0).sum().sum()
        print(f"  做多信号: {long_count}")
        print(f"  中性信号: {neutral_count}")
        
    except Exception as e:
        print(f"❌ 信号计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 展示每日信号 ============
    print("\n📋 最近 5 天信号示例:")
    for date in positions.index[-5:]:
        day_positions = positions.loc[date]
        longs = day_positions[day_positions > 0].index.tolist()
        if longs:
            print(f"  {date.strftime('%Y-%m-%d')}: 做多 {longs}")
        else:
            print(f"  {date.strftime('%Y-%m-%d')}: 无信号")
    
    # ============ 计算 IC 统计 ============
    print("\n📈 IC 统计 (近 30 天):")
    ic_stats = compute_ic_for_signal(positions, panel["close"], forward_period=5)
    print(f"  IC 均值: {ic_stats['ic_mean']:.4f}")
    print(f"  IC 标准差: {ic_stats['ic_std']:.4f}")
    print(f"  IR (信息比率): {ic_stats['ir']:.4f}")
    print(f"  IC 正比例: {ic_stats['ic_positive_ratio']:.2%}")
    
    # ============ 使用适配器生成信号 (兼容回测引擎) ============
    print("\n🔄 生成回测引擎兼容信号...")
    
    try:
        signal_dict = engine.generate(data_map)
        print(f"  生成信号数量: {len(signal_dict)}")
        
        for code, signal in signal_dict.items():
            non_zero = (signal != 0).sum()
            print(f"  {code}: {non_zero}/{len(signal)} 非零信号")
    except Exception as e:
        print(f"  ⚠️ 适配器失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


def generate_demo_data(codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """生成演示数据 (当数据源不可用时)"""
    import pandas as pd
    import numpy as np
    
    print("📝 生成演示数据...")
    
    dates = pd.date_range(start_date, end_date, freq="D")
    data_map = {}
    
    rng = np.random.default_rng(42)
    for i, code in enumerate(codes):
        # 模拟价格数据
        base_price = 50 + rng.integers(0, 100)
        returns = rng.standard_normal(len(dates)) * 0.02
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # 添加一些趋势
        trend = np.linspace(0, 0.1, len(dates))
        close_prices *= (1 + trend)
        
        df = pd.DataFrame({
            "open": close_prices * (1 + rng.uniform(-0.01, 0.01, len(dates))),
            "high": close_prices * (1 + rng.uniform(0, 0.02, len(dates))),
            "low": close_prices * (1 - rng.uniform(0, 0.02, len(dates))),
            "close": close_prices,
            "volume": rng.integers(1_000_000, 10_000_000, len(dates)),
        }, index=dates)
        
        data_map[code] = df
        print(f"  ✅ {code}: {len(df)} 条演示数据")
    
    return data_map


if __name__ == "__main__":
    main()