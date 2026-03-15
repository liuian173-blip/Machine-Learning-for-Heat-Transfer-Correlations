# Machine Learning for Heat Transfer Correlations
### A Self-Directed Engineering Learning Project (~17 hours)

---

## Project Overview

This project explores how machine learning models can approximate **heat transfer correlations** in ribbed microchannel geometries — a problem traditionally solved using empirical correlations or CFD simulations.

The goal is **not to build a production-level predictive model**, but to understand how a typical engineering ML workflow operates end-to-end:

- Physics-inspired synthetic dataset generation
- Data cleaning pipeline
- Regression model training and comparison
- Automated hyperparameter optimization
- Engineering-oriented interpretation of model results

The entire project was completed as a **personal learning exercise over approximately 17 hours across two days**.

---

## Project Motivation

Many engineering problems rely on **empirical correlations** that relate dimensionless parameters (Reynolds number, Prandtl number, geometry) to heat transfer performance. Recent research has explored using machine learning to **approximate or replace these correlations**.

This project attempts to reproduce a simplified version of that idea using a small synthetic dataset and interpretable models, grounded in four papers I identified and read independently:

- **Kwon et al. (2020)** provided the core ML-for-heat-transfer framing and the idea that geometric parameters alone can serve as predictive features
- **Tizakast et al. (2023)** demonstrated how ensemble and neural network methods compare on fluid dynamics problems
- **Weihing et al. (2014)** gave physical grounding for the ribbed channel setup, including how rib geometry influences heat transfer enhancement
- **Tsai et al. (2020)** provided the methodological basis for using simulated annealing as a hyperparameter optimization strategy for neural networks

---

## My Contributions

This project was designed and driven by me as a self-directed learning exercise.

**Project design**
- Identifying these four papers and defining a reproducible learning scope from them
- Defining the full project scope: 3-model pipeline (RF, SA-tuned MLP), physics-based data generation, engineering-oriented evaluation
- Writing the initial problem framing — physical parameters, dataset structure, modelling requirements — that set up the entire workflow
- Deciding the repository structure, file naming, and how scripts connect to each other
- Choosing to separate SA and NN into independent scripts rather than one combined file

**Dataset design**
- Constructing a synthetic dataset based on ribbed microchannel geometry
- Designing geometry-derived features: blockage ratio, hydraulic diameter, surface area ratio
- Deliberately injecting noise, missing values, and duplicates to create a realistic cleaning problem

**Active decisions during development**
- Reading the SA temperature decay curve and calculating that 100 iterations was sufficient (floor step ~95), reducing from 200
- Catching that `T_min=0.01` wasted 127 out of 200 iterations, and requesting the fix
- Noticing the neural network script was re-training an already-saved Random Forest, and requesting a `joblib`-based design instead
- Removing the second subplot from the SA visualisation because one chart was enough
- Choosing `test_size=0.25` for the SA validation split
- Deciding to suppress `ConvergenceWarning` after understanding what it meant and confirming it did not affect results

**Engineering interpretation**
- Connecting feature importance results back to physical mechanisms (Dh_mm dominance aligns with Kwon et al. and Weihing et al.)
- Evaluating model outputs with R², RMSE, and MAE in an engineering context

---

## AI Usage Disclosure

I used Claude (Anthropic) as a coding assistant and explanation tool throughout this project.

**AI assisted with:**
- Writing Python code based on my specifications
- Explaining concepts I asked about (cross-validation, SA acceptance criteria, warm_start, MLP, feature importance, etc.)
- Finding bugs I introduced (operator precedence in `h_conv`, missing `os.makedirs`, print order, `T0` overwritten by loop)
- Translating my design decisions into working code
- Constructing the semi-empirical Nu correlation used in data generation (**not my own derivation** — generated with AI assistance based on standard heat-transfer principles)

**The way I think about it:** I was the engineer making decisions. AI was the tool that implemented them and explained things I needed to learn along the way.

---

## Physics-Inspired Dataset Generation

To avoid running CFD simulations, a **synthetic dataset** was generated using a simplified semi-empirical heat transfer model. The Nu correlation was **constructed with AI assistance** based on:

- Baseline laminar Nusselt number (Shah & London, 1978: Nu₀ ≈ 5.385)
- Blockage ratio effects on flow obstruction
- Surface area enhancement due to ribs
- Geometric non-uniformity driving additional mixing

The objective was **not to produce a high-fidelity physical model**, but to generate a **physically reasonable dataset** suitable for testing an ML workflow.

---

## Physics Setup

| Parameter | Value |
|-----------|-------|
| Channel height H | 3.0 mm |
| Channel length L | 5.5 mm |
| Rib width | 0.5 mm |
| Rib spacing | 1.0 mm |
| Rib height candidates | {0.0001, 0.1, 1.0} mm |
| Reynolds number Re | 50 (laminar) |
| Prandtl number Pr | 7.0 (water ~20 °C) |
| Heat flux q" | 1 W/cm² |
| Total configurations | 3⁵ = 243 |

---

## Machine Learning Models

**Random Forest** — baseline model. Robust on small datasets, interpretable feature importance, no normalization required.

**Simulated Annealing** — automated hyperparameter search for the MLP, following the approach of Tsai et al. (2020). Probabilistically accepts worse solutions at high temperature to avoid local optima, then converges as temperature decreases. Search space: 10 architectures × 5 regularization values × 5 learning rates = 250 combinations. Results saved to JSON for the MLP script to load.

**Neural Network (MLP)** — feedforward network trained to capture nonlinear geometry–heat transfer relationships. Architecture and hyperparameters determined by SA search. Requires feature normalization.

---

## What I Learned

**Data and preprocessing**
- Physical boundary filtering must come before IQR filtering
- Derived features (BR_mean, Area_ratio, Dh_mm) carry more predictive power than raw rib heights — consistent with Weihing et al.'s findings on geometry-driven heat transfer
- How injecting controlled noise creates a realistic cleaning problem

**Random Forest**
- Tree-based models do not need feature normalization
- Feature importance in an engineering context: Dh_mm dominates (~0.35), directly mapping to convective intensity — consistent with Kwon et al.
- Why 5-fold cross-validation is more trustworthy than a single split on 240 rows

**Neural Networks (MLP)**
- Why normalization is essential and what breaks without it
- How `warm_start=True` with `max_iter=1` enables manual loss recording per epoch
- What ConvergenceWarning means and when suppressing it is the right call

**Simulated Annealing**
- Acceptance rule: always take improvements, probabilistically accept worse solutions at high temperature — following the SA-for-hyperparameter-tuning framework in Tsai et al. (2020)
- How to calculate effective exploration steps: `log(T_min / T0) / log(cooling)`
- How I caught that my original parameters were wasting more than half the iterations

**Software engineering habits**
- Create output directories before writing files (`os.makedirs(..., exist_ok=True)`)
- Save trained models with `joblib` so downstream scripts do not retrain them
- Store initial values in a separate variable before a loop modifies them

---

## Model Evaluation

| Metric | Description |
|--------|-------------|
| R² | Proportion of variance explained |
| RMSE | Root mean squared error [W/m²K] |
| MAE | Mean absolute error [W/m²K] |

Visualizations generated:

- Predicted vs. true value plots (train + test)
- Residual distribution
- Feature importance ranking
- MLP training loss curve
- SA temperature decay schedule

---

## Run Order

```bash
pip install numpy pandas scikit-learn matplotlib joblib

# Step 1 — generate raw dataset
python Data/Data_generation/Physics_based_dataset.py

# Step 2 — clean dataset
python Data/Data_cleaning/Data_clean.py

# Step 3 — train Random Forest, save model
python Models/random_forests.py

# Step 4 — SA hyperparameter search, save best params
python Models/stimulaed_annealing.py

# Step 5 — train MLP, compare with RF
python Models/neural_network.py
```

---

## Repository Structure

```
ml-heat-transfer-correlations/
├── Data/
│   ├── Data_cleaning/
│   │   ├── Data_clean.py
│   │   └── ML_Dataset_cleaned.csv         (generated)
│   └── Data_generation/
│       ├── Physics_based_dataset.py
│       └── ML_Dataset_heat_chaos.csv      (generated)
├── Figure_output/
│   ├── model_comparison.png
│   ├── nn_loss_curve.png
│   ├── nn_predicted_vs_true.png
│   ├── rf_feature_importance.png
│   ├── rf_predicted_vs_true.png
│   ├── rf_residual.png
│   └── sa_temperature_curve.png
├── Models/
│   ├── Model_Parameters/
│   │   ├── best_hyperparams.json          (generated)
│   │   └── random_forest.pkl              (generated)
│   ├── random_forests.py
│   ├── stimulaed_annealing.py
│   └── neural_network.py
└── README.md
```

---

## References

Kwon, B., Ejaz, F., & Hwang, L. K. (2020). Machine learning for heat transfer correlations. *International Communications in Heat and Mass Transfer*, 116, 104694. https://doi.org/10.1016/j.icheatmasstransfer.2020.104694

Tizakast, Y., Kaddiri, M., Lamsaadi, M., & Makayssi, T. (2023). Machine learning based algorithms for modeling natural convection fluid flow and heat and mass transfer in rectangular cavities filled with non-Newtonian fluids. *Engineering Applications of Artificial Intelligence*, 119, 105750. https://doi.org/10.1016/j.engappai.2022.105750

Tsai, C.-W., Hsia, C.-H., Yang, S.-J., Liu, S.-J., & Fang, Z.-Y. (2020). Optimizing hyperparameters of deep learning in predicting bus passengers based on simulated annealing. *Applied Soft Computing*, 88, 106068. https://doi.org/10.1016/j.asoc.2020.106068

Weihing, P., Younis, B. A., & Weigand, B. (2014). Heat transfer enhancement in a ribbed channel: Development of turbulence closures. *International Journal of Heat and Mass Transfer*, 76, 509–522. https://doi.org/10.1016/j.ijheatmasstransfer.2014.04.052

---
---

# 机器学习热传导关联式
### 自主学习项目（约 17 小时）

---

## 项目概述

本项目探索机器学习模型如何近似预测**带肋微通道中的传热关联式**——这类问题传统上通过经验公式或 CFD 仿真解决。

目标**不是构建工业级预测模型**，而是理解一个典型工程 ML 流程的完整运作方式：

- 基于物理启发的合成数据生成
- 数据清洗流程
- 回归模型训练与对比
- 自动化超参数优化
- 将模型结果与物理机制联系起来

整个项目作为**个人学习练习，在两天约 17 小时内完成**。

---

## 项目动机

工程领域大量问题依赖**经验关联式**，将无量纲参数（雷诺数、普朗特数、几何参数）与传热性能关联起来。近年来，有研究尝试用机器学习**近似或替代这些关联式**。

本项目尝试用小规模合成数据集和可解释模型对这一思想进行简化复现，基于我独立找到并阅读的四篇论文：

- **Kwon et al. (2020)**：提供 ML 用于传热的核心框架
- **Tizakast et al. (2023)**：展示了集成方法和神经网络在流体问题上的对比方式
- **Weihing et al. (2014)**：为带肋通道提供物理依据，包括肋条几何如何影响传热增强
- **Tsai et al. (2020)**：提供了使用模拟退火优化神经网络超参数的方法论依据

---

## 我的工作内容

**项目设计**
- 找到这四篇论文，确定学习方向并定义可复现的项目范围
- 定义完整项目范围：三模型流程（RF、SA 调参、MLP）、基于物理的数据生成、面向工程的评估
- 写出初始提示词，设定整个问题背景、物理参数、数据集结构和建模要求——这个框架决定了后续所有内容
- 决定仓库结构、文件命名，以及脚本之间如何连接
- 决定把 SA 和神经网络拆成两个独立文件

**数据集设计**
- 构建带肋微通道几何结构的合成数据集
- 设计几何派生特征：阻塞比、水力直径、表面积增强比
- 人为注入噪声、缺失值和重复行，构造真实的清洗问题

**开发过程中的主动决策**
- 读完温度衰减曲线后计算出 100 步够用，把迭代次数从 200 改为 100
- 发现 `T_min=0.01` 导致 200 次迭代中 127 次被浪费，要求修复
- 注意到神经网络脚本在重新训练已存在的随机森林，要求用 `joblib` 改成加载模型
- 删掉 SA 可视化的第二张子图
- 自己决定验证集比例 `test_size=0.25`
- 理解 ConvergenceWarning 的含义后决定忽略它

**工程解释**
- 将特征重要性结果联系回物理机制（Dh_mm 主导与 Kwon et al. 和 Weihing et al. 一致）

---

## AI 使用说明

我使用了 Claude（Anthropic）作为编程助手和概念解释工具。

**AI 协助的部分：**
- 根据我的要求编写 Python 代码
- 解释我主动提问的概念（交叉验证、SA 接受准则、warm_start、MLP、特征重要性等）
- 找出我引入的 bug（`h_conv` 运算符优先级、缺少 `os.makedirs`、print 顺序错误、`T0` 被循环覆盖）
- 把我的设计决策转化为可运行代码
- 构建数据生成中使用的半经验 Nu 关联式（**非我自己推导**，由 AI 参考标准传热原理生成）

**我的理解是：我是做决策的工程师，AI 是执行决策并在需要时解释概念的工具。**

---

## 基于物理启发的数据生成

为避免 CFD 仿真，使用**半经验模型生成合成数据集**。Nu 关联式在 AI 辅助下构建，参考了：

- 层流条件下的基准努塞尔数（Shah & London, 1978：Nu₀ ≈ 5.385）
- 肋条阻塞比效应
- 肋条增加的换热面积
- 几何不均匀性带来的额外混合

目标**不是精确物理预测**，而是生成**物理上合理的数据集**用于 ML 实验。

---

## 我学到了什么

**数据与预处理**
- 物理边界过滤必须在 IQR 过滤之前
- 派生特征比原始肋高携带更多预测信息，与 Weihing et al. 的几何驱动传热论据一致
- 注入可控噪声构造真实清洗问题的方法

**随机森林**
- 树模型不需要归一化
- Dh_mm 主导（约 0.35），与 Kwon et al. 的物理推理一致
- 5 折交叉验证比 240 行数据的单次划分更可靠

**神经网络（MLP）**
- 归一化的必要性
- `warm_start=True` + `max_iter=1` 手动记录每步 loss 的原理
- ConvergenceWarning 的含义和安全忽略的条件

**模拟退火**
- 接受准则：更好必接受，更差按温度概率接受——遵循 Tsai et al. (2020) 的 SA 超参数调优框架
- 有效探索步数的计算：`log(T_min / T0) / log(cooling)`
- 发现并修复参数浪费超过一半迭代次数的问题

**软件工程习惯**
- 写文件前创建目录
- 用 `joblib` 保存模型，避免重复训练
- 循环前保存初始变量值

---

## 参考文献

Kwon, B., Ejaz, F., & Hwang, L. K. (2020). Machine learning for heat transfer correlations. *International Communications in Heat and Mass Transfer*, 116, 104694. https://doi.org/10.1016/j.icheatmasstransfer.2020.104694

Tizakast, Y., Kaddiri, M., Lamsaadi, M., & Makayssi, T. (2023). Machine learning based algorithms for modeling natural convection fluid flow and heat and mass transfer in rectangular cavities filled with non-Newtonian fluids. *Engineering Applications of Artificial Intelligence*, 119, 105750. https://doi.org/10.1016/j.engappai.2022.105750

Tsai, C.-W., Hsia, C.-H., Yang, S.-J., Liu, S.-J., & Fang, Z.-Y. (2020). Optimizing hyperparameters of deep learning in predicting bus passengers based on simulated annealing. *Applied Soft Computing*, 88, 106068. https://doi.org/10.1016/j.asoc.2020.106068

Weihing, P., Younis, B. A., & Weigand, B. (2014). Heat transfer enhancement in a ribbed channel: Development of turbulence closures. *International Journal of Heat and Mass Transfer*, 76, 509–522. https://doi.org/10.1016/j.ijheatmasstransfer.2014.04.052

---

## 运行顺序

```bash
pip install numpy pandas scikit-learn matplotlib joblib

python Data/Data_generation/Physics_based_dataset.py
python Data/Data_cleaning/Data_clean.py
python Models/random_forests.py
python Models/stimulaed_annealing.py
python Models/neural_network.py
```