import os
import pandas as pd
import numpy as np

# 创建输出目录（如果不存在则新建，exist_ok=True 避免重复创建时报错）
os.makedirs('Data/Data_cleaning', exist_ok=True)

# STEP 1：读入数据，基础检查

df = pd.read_csv('Data/Data_generation/ML_Dataset_heat_chaos.csv')  # 跳过第一行（如果有重复标题）

print("原始数据形状")
print(df.shape)               # 应该是 (243, 13)

print("\n各列数据类型")
print(df.dtypes)              # 检查有没有列变成 object（字符串）

print("\n缺失值数量")
print(df.isnull().sum())      # 每列有几个 NaN

print("\n重复行数量")
print(df.duplicated().sum())  # 有几行是完全重复的

# STEP 2 清洗

# 第一步：先删重复行 
df = df.drop_duplicates()
print(f"删除重复行后：{len(df)} 行")

# 第二步：先用物理边界删异常值
# e1~e5 的合法值只有三个：0.0001, 0.1, 1.0
# 所以范围必须在 [0, 2] 之间，超出就是异常
valid_h = [0.0001, 0.1, 1.0]
for col in ['e1_mm','e2_mm','e3_mm','e4_mm','e5_mm']:
    before = len(df)
    df = df[df[col].isin(valid_h) | df[col].isna()]  # 保留合法值或缺失值
    print(f"{col} 异常值删除：{before - len(df)} 行")

# Nu 物理边界（公式推导最大值约20，上界收紧至25以覆盖极端情况）
before = len(df)
df = df[(df['Nu'] >= 5.0) & (df['Nu'] <= 25.0) | df['Nu'].isna()]
print(f"Nu 物理边界删除：{before - len(df)} 行")

# 派生特征的物理边界（数据生成脚本同样注入了极值，需要一并清除）
# BR_mean / BR_max 是阻塞比，物理范围 [0, 1]
# Dh_mm 是水力直径，对应最小肋高时约为 2*(H-0.0001)≈6 mm，最大不超过 2*H=6 mm
# Area_ratio 是面积增强比，恒 >= 1，最大不超过 3
derived_bounds = {
    'BR_mean':    (0.0, 1.0),
    'BR_max':     (0.0, 1.0),
    'BR_std':     (0.0, 1.0),
    'Dh_mm':      (0.0, 6.1),
    'Area_ratio': (1.0, 3.0),
}
for col, (lo, hi) in derived_bounds.items():
    before = len(df)
    df = df[(df[col] >= lo) & (df[col] <= hi) | df[col].isna()]
    print(f"{col} 物理边界删除：{before - len(df)} 行")

#  第三步：再删缺失值 
before = len(df)
df = df.dropna()
print(f"删除缺失值后：{len(df)} 行")

#  第四步：IQR 检测 h_conv 的异常值 
Q1_h = df['h_conv'].quantile(0.25)
Q3_h = df['h_conv'].quantile(0.75)
IQR_h = Q3_h - Q1_h
lower_h = Q1_h - 1.5 * IQR_h
upper_h = Q3_h + 1.5 * IQR_h
before = len(df)
df = df[(df['h_conv'] >= lower_h) & (df['h_conv'] <= upper_h)]
print(f"h_conv IQR 过滤后：{len(df)} 行，删除了 {before - len(df)} 行")

# 第五步：保存清洗后的 CSV 
df.to_csv('Data/Data_cleaning/ML_Dataset_cleaned.csv', index=False)
print(f"\n✅ 清洗后数据已保存：ML_Dataset_cleaned.csv")
print(f"   最终行数：{len(df)}（期望接近243）")