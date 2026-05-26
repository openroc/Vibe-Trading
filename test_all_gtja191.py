#!/usr/bin/env python3
"""
测试所有 GTJA191 Alpha 因子
"""
import sys
sys.path.insert(0, 'agent')
sys.path.insert(0, '.')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.factors.registry import get_default_registry

# 数据加载
from backtest.loaders.tdx_loader import DataLoader as TDXLoader

# 沪深300 部分股票
CSI300_CODES = [
    '000001.SZ', '000002.SZ', '000063.SZ', '000100.SZ', '000333.SZ',
    '000338.SZ', '000651.SZ', '000661.SZ', '000858.SZ', '000876.SZ',
    '002001.SZ', '002027.SZ', '002046.SZ', '002049.SZ', '002129.SZ',
    '002142.SZ', '002230.SZ', '002236.SZ', '002241.SZ', '002252.SZ',
    '002304.SZ', '002311.SZ', '002352.SZ', '002371.SZ', '002415.SZ',
    '002460.SZ', '002475.SZ', '002493.SZ', '002594.SZ', '002601.SZ',
    '002602.SZ', '002607.SZ', '002714.SZ', '002736.SZ', '002812.SZ',
    '300014.SZ', '300059.SZ', '300122.SZ', '300274.SZ', '300750.SZ',
]

def load_data(codes, start_date, end_date):
    """加载数据"""
    loader = TDXLoader()
    data_map = {}
    batch_size = 20
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        try:
            batch_data = loader.fetch(batch, start_date, end_date, 
                                      fields=['Open', 'High', 'Low', 'Close', 'Volume', 'Amount'])
            data_map.update(batch_data)
        except Exception as e:
            print(f"  批次 {i//batch_size + 1} 错误: {e}")
    return data_map

def build_panel(data_map):
    """构建宽面板"""
    all_dates = set()
    for df in data_map.values():
        all_dates.update(df.index)
    
    dates = sorted(all_dates)
    
    panel = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        panel[field] = pd.DataFrame(index=dates, columns=list(data_map.keys()))
        for code, df in data_map.items():
            if field in df.columns:
                panel[field][code] = df[field]
        panel[field] = panel[field].astype(float)
    
    return panel

def calc_ic(panel, alpha_signals, forward_returns):
    """计算 IC"""
    valid_idx = alpha_signals.notna() & forward_returns.notna()
    if valid_idx.sum() < 10:
        return np.nan
    
    ic = alpha_signals[valid_idx].corrwith(forward_returns[valid_idx])
    return ic.iloc[0] if len(ic) > 0 else np.nan

def test_alpha(alpha_id, panel, registry):
    """测试单个因子"""
    try:
        signals = registry.compute(alpha_id, panel)
        
        if signals is None or signals.empty:
            return None
        
        # 移除全是 NaN 的信号
        if signals.notna().sum().sum() < 10:
            return None
        
        close = panel['close']
        forward_ret = close.pct_change().shift(-1)
        
        ic_values = []
        for date in signals.index:
            if date not in close.index or date not in forward_ret.index:
                continue
            
            sig = signals.loc[date]
            fwd = forward_ret.loc[date]
            
            valid = sig.notna() & fwd.notna()
            if valid.sum() < 5:
                continue
            
            ic = sig[valid].corr(fwd[valid])
            if not np.isnan(ic):
                ic_values.append(ic)
        
        if len(ic_values) < 20:
            return None
        
        ic_mean = np.mean(ic_values)
        ic_std = np.std(ic_values)
        ic_plus_rate = np.mean([ic > 0 for ic in ic_values])
        ir = ic_mean / ic_std if ic_std > 0 else 0
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ic_plus_rate': ic_plus_rate,
            'n_samples': len(ic_values),
            'nan_ratio': signals.isna().mean().mean(),
        }
    except Exception as e:
        return None

def main():
    print("=" * 70)
    print("GTJA191 Alpha 因子批量测试")
    print("=" * 70)
    
    # 配置
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    
    print(f"\n[配置]")
    print(f"  股票池: 沪深300 ({len(CSI300_CODES)} 只)")
    print(f"  时间范围: {start_date} ~ {end_date}")
    
    # 加载数据
    print(f"\n加载数据...")
    data_map = load_data(CSI300_CODES, start_date, end_date)
    print(f"  成功加载: {len(data_map)} 只股票")
    
    # 构建面板
    print(f"\n构建面板...")
    panel = build_panel(data_map)
    print(f"  面板形状: {panel['close'].shape}")
    
    # 获取 registry
    registry = get_default_registry()
    
    # 测试所有 GTJA191 因子
    print(f"\n测试 GTJA191 因子...")
    results = []
    
    for i in range(1, 192):
        alpha_id = f"gtja191_{i:03d}"
        try:
            result = test_alpha(alpha_id, panel, registry)
            if result:
                result['alpha_id'] = alpha_id
                results.append(result)
                
                # 实时显示优质因子
                if result['ir'] > 0.3 and result['ic_plus_rate'] > 0.55:
                    print(f"  ✅ {alpha_id}: IC={result['ic_mean']:.4f}, IR={result['ir']:.4f}, IC+={result['ic_plus_rate']*100:.1f}%")
        except Exception as e:
            pass
        
        if i % 50 == 0:
            print(f"  进度: {i}/191")
    
    print(f"\n{'='*70}")
    print(f"测试完成! 有效因子: {len(results)}/191")
    print("=" * 70)
    
    # 按 IR 排序
    results.sort(key=lambda x: x['ir'], reverse=True)
    
    # 显示 Top 20 因子
    print(f"\n📊 Top 20 因子 (按 IR 排序):")
    print(f"{'排名':<4} {'因子ID':<15} {'IC':<10} {'IR':<10} {'IC+率':<10} {'NaN%':<10}")
    print("-" * 60)
    
    for i, r in enumerate(results[:20], 1):
        status = "✅" if r['ir'] > 0.3 and r['ic_plus_rate'] > 0.55 else "⚠️ "
        print(f"{status}{i:<3} {r['alpha_id']:<15} {r['ic_mean']:.4f}    {r['ir']:.4f}    {r['ic_plus_rate']*100:.1f}%     {r['nan_ratio']*100:.1f}%")
    
    # 统计
    print(f"\n📈 统计汇总:")
    good_factors = [r for r in results if r['ir'] > 0.3 and r['ic_plus_rate'] > 0.55]
    print(f"  优质因子 (IR>0.3, IC+>55%): {len(good_factors)} 个")
    
    avg_ic = np.mean([r['ic_mean'] for r in results])
    avg_ir = np.mean([r['ir'] for r in results])
    print(f"  平均 IC: {avg_ic:.4f}")
    print(f"  平均 IR: {avg_ir:.4f}")
    
    # 保存结果
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('ir', ascending=False)
    df_results.to_csv('alpha_test_results.csv', index=False)
    print(f"\n💾 结果已保存到 alpha_test_results.csv")

if __name__ == '__main__':
    main()