"""
数据生成脚本：基于论文几何参数重建热传导ML数据集
论文设定：
  - 二维沟槽通道，含5根肋条
  - 沟槽高度 H = 3 mm，总长 L = 5.5 mm
  - 肋宽 w = 0.5 mm，肋间距 p = 1 mm
  - 肋高候选值：{0.0001, 0.1, 1.0} mm
  - Re = 50（层流），冷却剂：水，Pr ≈ 7
  - 底壁恒定热通量 q" = 1 W/cm²，顶壁绝热
"""

# 导入必要的库
import numpy as np
import pandas as pd
import itertools
import os

# 创建输出目录（如果不存在则新建，exist_ok=True 避免重复创建时报错）
os.makedirs('Data/Data_generation', exist_ok=True)

# 1. 通道几何参数（固定）
H      = 3.0      # 沟槽高度 [mm]
L      = 5.5      # 通道总长 [mm]
w_rib  = 0.5      # 肋宽 [mm]
p_rib  = 1.0      # 肋间距 [mm]
N_ribs = 5        # 肋条数量

Re     = 50       # 雷诺数（层流区，Re < 2300）
Pr     = 7.0      # 普朗特数（水，约20°C）
q_flux = 1.0      # 热通量 [W/cm²]

# 肋高的三个候选值 [mm] - 用于生成不同的几何配置, 总数据量为 (3)n^5 种组合
h_choices = [0.0001, 0.1, 1.0]

# 2. 生成所有 3^5 = 243 种组合
# 使用itertools.product生成所有可能的肋高组合
# 每根肋有3种高度选择，5根肋，总共3^5=243种不同配置
all_combos = list(itertools.product(h_choices, repeat=N_ribs))
print(f"总组合数：{len(all_combos)}（应为 3^5 = {3**5}）")

# 初始化存储每种组合数据的列表
rows = []

# 遍历每种肋高组合
for combo in all_combos:
    # 解包组合中的5个肋高值
    e1, e2, e3, e4, e5 = combo
    # 将肋高转换为numpy数组，便于后续计算
    rib_heights = np.array([e1, e2, e3, e4, e5])  # [mm]
    
    # 阻塞比（Blockage Ratio）= 肋高 / 通道高
    # 表示肋条对通道截面的阻塞程度，无量纲
    BR = rib_heights / H                            # 各肋阻塞比，无量纲
    BR_mean = BR.mean()                             # 平均阻塞比
    BR_max  = BR.max()                              # 最大阻塞比
    BR_std  = BR.std()                              # 阻塞比的标准差，表示非均匀度

    # 有效水力直径：近似为矩形通道
    # 截面积近似 A = H（单位宽度），考虑肋占据部分面积
    # 这里用平均有效通道高度计算
    h_eff_mean = H - np.mean(rib_heights)           # 有效平均通道高 [mm]
    Dh = 2 * h_eff_mean                             # 水力直径（2D通道近似）[mm]

    # 表面增强比：有肋时的湿周 / 光滑时的湿周
    # 每根肋增加 2×h 的侧面积（上下两侧）
    wetted_plain = 2 * L                            # 光滑通道的湿周长（底+顶）[mm]
    wetted_ribs  = 2 * np.sum(rib_heights)          # 肋侧面积增量 [mm]
    Area_ratio   = 1 + wetted_ribs / wetted_plain   # 表面积比（增强比）

    # 层流恒热通量平行板基础值：Nu₀ ≈ 5.385（Shah & London, 1978）
    Nu0 = 5.385
    # 肋条通过以下机制增强换热：
    #   1. 增大换热面积（Area_ratio）
    #   2. 产生局部流动扰动（与BR正相关）
    #   3. 非均匀排布产生额外混合（与BR_std正相关）
    #
    # 半经验公式（参考 Promvonge & Thianpong, 2008 思路）：
    #   Nu = Nu₀ × (1 + α×BR_mean^β) × Area_ratio^γ × (1 + δ×BR_std)
    # 假设参数：α=2, β=0.5, γ=0.8, δ=1 (可根据数据调整)
    alpha, beta, gamma, delta = 2, 0.5, 0.8, 1
    Nu = Nu0 * (1 + alpha * BR_mean**beta) * Area_ratio**gamma * (1 + delta * BR_std)

    # 热阻 R_th = 1 / (h * A) ，但这里简化计算基于Nu
    # 假设特征长度为Dh，热导率k=0.6 W/mK (水)
    k = 0.6  # W/mK
    h_conv = Nu * k / (Dh / 1000)  # 换热系数 [W/m²K] (Dh mm 转 m，括号明确运算顺序)
    R_th = 1 / (h_conv * (L * 0.001) * 0.001)  # 热阻 [K/W] (简化，单位宽度)

    # 存储数据行
    row = {
        'e1_mm': e1, 'e2_mm': e2, 'e3_mm': e3, 'e4_mm': e4, 'e5_mm': e5,
        'BR_mean': BR_mean, 'BR_max': BR_max, 'BR_std': BR_std,
        'Dh_mm': Dh, 'Area_ratio': Area_ratio,
        'Nu': Nu, 'R_th': R_th, 'h_conv': h_conv
    }
    rows.append(row)

# 将数据转换为pandas DataFrame
df = pd.DataFrame(rows)

# === 添加数据混乱：重复行、缺失值、不合理值 ===
# 设置随机种子以确保可重现性
np.random.seed(42)

# 1. 添加重复行：随机复制10%的行
n_duplicates = int(0.1 * len(df))
duplicate_indices = np.random.choice(df.index, n_duplicates, replace=True)
df_duplicates = df.loc[duplicate_indices].copy()
df = pd.concat([df, df_duplicates], ignore_index=True)

# 2. 添加缺失值：随机在数值列中设为NaN（约5%的单元格）
numeric_cols = ['e1_mm', 'e2_mm', 'e3_mm', 'e4_mm', 'e5_mm', 'BR_mean', 'BR_max', 'BR_std', 'Dh_mm', 'Area_ratio', 'Nu', 'R_th', 'h_conv']
for col in numeric_cols:
    mask = np.random.rand(len(df)) < 0.01  # 1% 概率
    df.loc[mask, col] = np.nan

# 3. 添加异常值：随机在数值列中放入负数或极值（约3%的单元格）
for col in numeric_cols:
    mask = np.random.rand(len(df)) < 0.015   # 1.5% 概率
    unreasonable_values = np.random.choice([-999, -100, 1e6, -1e6], size=mask.sum())  # 明显不合理的值
    df.loc[mask, col] = unreasonable_values

# 保存为CSV文件（不包含索引列），保存在当前文件夹中
df.to_csv('Data/Data_generation/ML_Dataset_heat_chaos.csv', index=False)  # 修正路径为相对路径

# 输出完成信息和数据集统计
print("\n✅ 数据集生成完成！")
print(f"   形状: {df.shape}  →  {df.shape[0]}行 × {df.shape[1]}列")
print(f"\n── 前5行预览 ──")
print(df.head().to_string())
print(f"\n── 统计摘要 ──")
print(df[['BR_mean','BR_max','Area_ratio','Nu','R_th','h_conv']].describe().round(4).to_string())