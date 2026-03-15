"""
神经网络训练脚本（MLP）
依赖：models/best_hyperparams.json（由 simulated_annealing.py 生成）
      models/random_forest.pkl    （由 random_forests.py 生成）
输出：三张图 + 终端评估指标
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

#  读取 SA 搜索得到的最优超参数 
with open('Models/Model_parameters/best_hyperparams.json', 'r') as f:
    hp = json.load(f)

print("  载入超参数（来自模拟退火搜索结果）")
print(f"  hidden_layer_sizes : {tuple(hp['hidden_layer_sizes'])}")
print(f"  alpha              : {hp['alpha']}")
print(f"  learning_rate_init : {hp['learning_rate_init']}")
print(f"  SA 验证集 RMSE     : {hp['val_rmse']:.4f} W/m2K")

#  读取数据 
df = pd.read_csv('Data/Data_cleaning/ML_Dataset_cleaned.csv')

feature_cols = ['e1_mm','e2_mm','e3_mm','e4_mm','e5_mm',
                'BR_mean','BR_max','BR_std','Dh_mm','Area_ratio']
target_col   = 'h_conv'

X = df[feature_cols].values
y = df[target_col].values

# 神经网络必须归一化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# MLP 用归一化数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# 随机森林用原始数据划分 25% 测试集
X_train_raw, X_test_raw, _, _ = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 训练最终 MLP 
# warm_start=True + max_iter=1：每次调用 fit 只训练1个 epoch
# 目的：手动记录每步 loss，用于绘制训练曲线
EPOCHS = 400  # 400 epochs，过多可能过拟合

final_model = MLPRegressor(
    hidden_layer_sizes=tuple(hp['hidden_layer_sizes']),
    alpha=hp['alpha'],
    learning_rate_init=hp['learning_rate_init'],
    max_iter=1,
    warm_start=True,       # 保留上次权重，继续接着训练
    random_state=42,
    n_iter_no_change=500,  # 关闭内置早停，由 EPOCHS 控制
)

train_loss = []
for _ in range(EPOCHS):
    final_model.fit(X_train, y_train)
    train_loss.append(final_model.loss_)

# 评估指标 
y_pred       = final_model.predict(X_test)
y_pred_train = final_model.predict(X_train)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

# 5折交叉验证（最终报告指标）
cv_r2 = cross_val_score(final_model, X_scaled, y, cv=5, scoring='r2')

print(f"\n 神经网络（MLP）评估结果")
print(f"R2             = {r2:.4f}")
print(f"RMSE           = {rmse:.4f}  W/m2K")
print(f"MAE            = {mae:.4f}  W/m2K")
print(f"5折 CV R2      = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# 随机森林基线（直接加载已训练好的模型，无需重新训练）
rf        = joblib.load('models/random_forest.pkl')
y_pred_rf = rf.predict(X_test_raw)

rf_r2   = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae  = mean_absolute_error(y_test, y_pred_rf)

print(f"\n{'指标':<10} {'随机森林':>12} {'MLP (SA)':>12}")
print(f"{'R2':<10} {rf_r2:>12.4f} {r2:>12.4f}")
print(f"{'RMSE':<10} {rf_rmse:>12.4f} {rmse:>12.4f}")
print(f"{'MAE':<10} {rf_mae:>12.4f} {mae:>12.4f}")

# 图1：训练损失曲线（Loss vs Epoch）
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(range(1, EPOCHS + 1), train_loss, color='steelblue', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
ax1.set_title(
    f'MLP Training Loss Curve\n'
    f'hidden={tuple(hp["hidden_layer_sizes"])}  '
    f'alpha={hp["alpha"]}  lr={hp["learning_rate_init"]}',
    fontsize=11
)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_output/nn_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 图2：预测值 vs 真实值
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(y_train, y_pred_train,
            alpha=0.6, color='steelblue', edgecolors='white',
            s=50, label='Train set')
ax2.scatter(y_test, y_pred,
            alpha=0.9, color='tomato', edgecolors='white',
            s=60, label='Test set')

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())
ax2.plot([min_val, max_val], [min_val, max_val],
         'k--', linewidth=1.5, label='Perfect fit')

ax2.set_xlabel('True h_conv [W/m2K]', fontsize=12)
ax2.set_ylabel('Predicted h_conv [W/m2K]', fontsize=12)
ax2.set_title(f'MLP: Predicted vs True\nR2={r2:.4f}  RMSE={rmse:.4f}', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_output/nn_predicted_vs_true.png', dpi=150, bbox_inches='tight')
plt.close()

# 图3：随机森林 vs MLP 三指标对比条形图
metrics = ['R2', 'RMSE [W/m2K]', 'MAE [W/m2K]']
rf_vals = [rf_r2,  rf_rmse,  rf_mae]
nn_vals = [r2,     rmse,     mae]

fig3, axes = plt.subplots(1, 3, figsize=(11, 4))
colors = ['steelblue', 'tomato']

for ax, metric, rv, nv in zip(axes, metrics, rf_vals, nn_vals):
    bars = ax.bar(['Random\nForest', 'MLP\n(SA-tuned)'], [rv, nv],
                  color=colors, edgecolor='white', alpha=0.85, width=0.5)
    for bar, val in zip(bars, [rv, nv]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_title(metric, fontsize=12)
    ax.set_ylim(0, max(rv, nv) * 1.25)
    ax.grid(alpha=0.3, axis='y')

fig3.suptitle('Model Comparison: Random Forest vs MLP (SA-tuned)',
              fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('Figure_output/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ 三张图已保存至 Figure_output/")
print("   nn_loss_curve.png")
print("   nn_predicted_vs_true.png")
print("   model_comparison.png")