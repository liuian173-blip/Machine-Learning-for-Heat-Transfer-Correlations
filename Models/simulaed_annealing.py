"""
模拟退火超参数搜索脚本
输出：最优超参数组合，保存至 models/best_hyperparams.json
      温度衰减曲线图，保存至 Figure_output/sa_temperature_curve.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

#  读取数据 
df = pd.read_csv('Data/Data_cleaning/ML_Dataset_cleaned.csv')

feature_cols = ['e1_mm','e2_mm','e3_mm','e4_mm','e5_mm',
                'BR_mean','BR_max','BR_std','Dh_mm','Area_ratio']
target_col   = 'h_conv'

X = df[feature_cols].values
y = df[target_col].values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 划分训练集和验证集
# 固定种子保证每次搜索在相同数据上进行
# 75% 训练集（用于每次评估时训练模型），25% 验证集（用于计算 RMSE 作为能量函数）
X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

#  超参数搜索空间 
# hidden_layer_sizes：网络结构，单层或双层，层数越多越复杂
HIDDEN_OPTIONS = [
    (32,), (64,), (128,),
    (32, 32), (64, 32), (64, 64), (128, 64), (128, 128),
    (64, 32, 16), (128, 64, 32),
]
ALPHA_OPTIONS  = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]   # 正则化强度，越大越简单，防止过拟合
LR_OPTIONS     = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # 初始学习率，越大收敛越快但可能不稳定，越小更稳定但训练慢

#  SA 辅助函数 
def random_state():
    return {
        'hidden': HIDDEN_OPTIONS[np.random.randint(len(HIDDEN_OPTIONS))],
        'alpha':  float(np.random.choice(ALPHA_OPTIONS)),
        'lr':     float(np.random.choice(LR_OPTIONS)),
    }

def neighbor_state(state):
    """随机改变一个超参数维度，生成邻居状态"""
    new = state.copy()
    dim = np.random.choice(['hidden', 'alpha', 'lr'])
    if dim == 'hidden':
        new['hidden'] = HIDDEN_OPTIONS[np.random.randint(len(HIDDEN_OPTIONS))]
    elif dim == 'alpha':
        new['alpha'] = float(np.random.choice(ALPHA_OPTIONS))
    else:
        new['lr'] = float(np.random.choice(LR_OPTIONS))
    return new

def evaluate(state):
    """
    在验证集上计算 RMSE，作为 SA 的能量函数
    max_iter=200 保证每次评估充分收敛，避免返回噪声 RMSE
    """
    model = MLPRegressor(
        hidden_layer_sizes=state['hidden'],
        alpha=state['alpha'],
        learning_rate_init=state['lr'],
        max_iter=200,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# 模拟退火主循环
#
# 核心逻辑：
#   当前状态能量更低 → 直接接受
#   当前状态能量更高 → 以 exp(-ΔE/T) 概率接受（温度越高概率越大）
#   温度 T 随迭代指数下降，前期广泛探索，后期集中收敛

T0       = 2.0    # 保存初始温度，用于图标题（T 会在循环中被修改）
T        = 2.0    # 初始温度，过高会接受过多差解，过低可能过早收敛到局部最优
T_min    = 0.002  # 终止温度，配合 cooling=0.93 在第 ~95 步触底，100步内有效覆盖
cooling  = 0.93   # 冷却率，触底步数 ≈ ln(T_min/T₀) / ln(cooling) ≈ 95 步
n_iter   = 150    # 总迭代次数

np.random.seed(42)
current_state  = random_state()
current_energy = evaluate(current_state)
best_state     = current_state.copy()
best_energy    = current_energy

print(f"SA 搜索开始 | 搜索空间：{len(HIDDEN_OPTIONS)}×{len(ALPHA_OPTIONS)}×{len(LR_OPTIONS)} 种组合")
print(f"初始 RMSE = {current_energy:.4f}")
print("-" * 55)

accept_count   = 0    # 记录接受次数，用于最终统计
temp_history   = []   # 记录每步温度，用于绘制衰减曲线

for i in range(n_iter):
    new_state  = neighbor_state(current_state)
    new_energy = evaluate(new_state)
    delta_E    = new_energy - current_energy

    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
        current_state  = new_state
        current_energy = new_energy
        accept_count  += 1

    if current_energy < best_energy:
        best_energy = current_energy
        best_state  = current_state.copy()

    temp_history.append(T)
    T = max(T * cooling, T_min)

    # 每 50 步打印一次进度
    if (i + 1) % 50 == 0:
        print(f"  iter {i+1:3d}/{n_iter}  T={T:.4f}  "
              f"best RMSE = {best_energy:.4f}")

print("-" * 55)
print(f"SA 搜索完成 | 接受率 {accept_count/n_iter*100:.1f}%")

#  打印最优超参数（核心输出）
print("       最优超参数组合")
print(f"  网络结构 hidden_layer_sizes : {best_state['hidden']}")
print(f"  正则化强度 alpha            : {best_state['alpha']}")
print(f"  初始学习率 learning_rate    : {best_state['lr']}")
print(f"  验证集 RMSE                 : {best_energy:.4f} W/m²K")

#  保存超参数到 JSON，供神经网络脚本读取 
result = {
    'hidden_layer_sizes': list(best_state['hidden']),
    'alpha':              best_state['alpha'],
    'learning_rate_init': best_state['lr'],
    'val_rmse':           float(best_energy),
}
with open('Models/Model_parameters/best_hyperparams.json', 'w') as f:
    json.dump(result, f, indent=4)

print("\n✅ 超参数已保存至 Models/Model_parameters/best_hyperparams.json")

# 可视化：温度衰减曲线
# 温度随迭代步数的指数衰减，标注 T_min 触底位置
iters = range(1, n_iter + 1)

plt.figure(figsize=(8, 4))

plt.plot(iters, temp_history, color='steelblue', linewidth=1.8, label='Temperature T')
plt.axhline(T_min, color='tomato', linestyle='--', linewidth=1.2, label=f'T_min = {T_min}')

# 标注温度实际触底的步数
floor_step = next((i + 1 for i, t in enumerate(temp_history) if t <= T_min + 1e-9), n_iter)
plt.axvline(floor_step, color='tomato', linestyle=':', linewidth=1.0, alpha=0.7)
plt.text(floor_step + 1, T_min + 0.05,
         f'Floor @ step {floor_step}', fontsize=9, color='tomato')

plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Temperature T', fontsize=11)
plt.title(f'Simulated Annealing Schedule\n'
          f'T0={T0}  T_min={T_min}  cooling={cooling}  n_iter={n_iter}',
          fontsize=11)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_output/sa_temperature_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 温度衰减曲线已保存至 Figure_output/sa_temperature_curve.png")

