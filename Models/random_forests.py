import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# 创建图片输出目录
os.makedirs('Models', exist_ok=True)
os.makedirs('Figure_output', exist_ok=True)
os.makedirs('Models/Model_parameters', exist_ok=True)

# 读入清洗后的数据
df = pd.read_csv('Data/Data_cleaning/ML_Dataset_cleaned.csv')

# 定义输入特征 和 目标变量
feature_cols = ['e1_mm','e2_mm','e3_mm','e4_mm','e5_mm',
                'BR_mean','BR_max','BR_std',
                'Dh_mm','Area_ratio']  # 输入特征列
target_col = 'h_conv'  # 目标变量列

X = df[feature_cols].values
y = df[target_col].values

# 划分训练集75% 测试集25%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42    # 固定随机种子，结果可重现
)

# 创建随机森林模型
rf_model = RandomForestRegressor(
    n_estimators=100, # 树的数量，越多越稳定但越慢
    max_depth=3,   # 树的最大深度，None表示直到叶子节点，越大越容易过拟合，3就足够了
    random_state=42   # 固定种子，结果可重现
)

# 训练（随机森林不需要归一化，直接用原始X）
rf_model.fit(X_train, y_train)

## 用测试集预测
y_pred = rf_model.predict(X_test)

# 计算三个评估指标
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"\n=== 随机森林评估结果 ===")
print(f"R²   = {r2:.4f}    （拟合程度。越接近1越好）")
print(f"RMSE = {rmse:.4f}  （均方误差。越小越好，单位和Nu一样）")
print(f"MAE  = {mae:.4f}   （平均绝对误差。越小越好）")

# 5折交叉验证（数据量小时比单次划分更可靠，减少 random_state 带来的偶然性）
cv_r2 = cross_val_score(rf_model, X, y, cv=5, scoring='r2') #轮流使用其中四分作为训练集，一分作为验证集，评估R²得分。返回5个得分值。
print(f"\n=== 5折交叉验证 R² ===")
print(f"各折: {cv_r2.round(4)}")
print(f"均值 ± 标准差: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

joblib.dump(rf_model, 'Models/Model_parameters/random_forest.pkl')
print("✅ 随机森林模型已保存至 Models/Model_parameters/random_forest.pkl")

# ── 图1：预测值 vs 真实值（训练集+测试集都画出来）──────
y_pred_train = rf_model.predict(X_train)  # 训练集的预测值

fig1, ax1 = plt.subplots(figsize=(6, 6))

ax1.scatter(y_train, y_pred_train,
            alpha=0.6, color='steelblue', edgecolors='white',
            s=50, label='Train set')
ax1.scatter(y_test, y_pred,
            alpha=0.9, color='tomato', edgecolors='white',
            s=60, label='Test set')

min_val = min(y_train.min(), y_test.min(), y_pred_train.min(), y_pred.min())
max_val = max(y_train.max(), y_test.max(), y_pred_train.max(), y_pred.max())

ax1.plot([min_val, max_val], [min_val, max_val],
         'k--', linewidth=1.5, label='Perfect fit')

ax1.set_xlabel('True h_conv [W/m²K]', fontsize=12)
ax1.set_ylabel('Predicted h_conv [W/m²K]', fontsize=12)
ax1.set_title(f'Predicted vs True\nR²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_output/rf_predicted_vs_true.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 图2：残差分布 ───────────────────────────────────
residuals = y_test - y_pred

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.hist(residuals, bins=20, color='salmon', edgecolor='white', alpha=0.85)
ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero error')
ax2.axvline(residuals.mean(), color='navy', linestyle='--', linewidth=1.5,
            label=f'Mean={residuals.mean():.2f}')

ax2.set_xlabel('Residual  (True - Predicted)  [W/m²K]', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Residual Distribution\n(ideal = symmetric around 0)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_output/rf_residual.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 图3：特征重要性 ─────────────────────────────────
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
sorted_features    = [feature_cols[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

fig3, ax3 = plt.subplots(figsize=(7, 5))
bars = ax3.bar(range(len(feature_cols)), sorted_importances,
               color='mediumseagreen', edgecolor='white', alpha=0.85)

# 在每根柱子顶部标注数值
for bar, val in zip(bars, sorted_importances):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax3.set_xticks(range(len(feature_cols)))
ax3.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Importance Score', fontsize=12)
ax3.set_title('Feature Importance\n(contribution to predicting h_conv)', fontsize=12)
ax3.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Figure_output/rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()