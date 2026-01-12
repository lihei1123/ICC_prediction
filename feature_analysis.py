import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# 设置图表样式
plt.style.use('default')
sns.set_palette("husl")

print("=== 特征选择分析开始 ===")

# 读取预处理后的数据
X_scaled = pd.read_csv('X_scaled.csv')
y_os = pd.read_csv('y_os.csv', header=None, names=['OS'])['OS']
y_binary = pd.read_csv('y_binary.csv', header=None, names=['OS_binary'])['OS_binary']

print(f"数据形状检查:")
print(f"X_scaled: {X_scaled.shape}")
print(f"y_os: {y_os.shape}")
print(f"y_binary: {y_binary.shape}")

# 读取特征名称
with open('feature_names.txt', 'r', encoding='utf-8') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"特征数量: {len(feature_names)}")

# 方法1: 单因素F值分析
print("\n=== 方法1: 单因素F值分析 ===")
f_scores, f_pvalues = f_regression(X_scaled, y_os)

# 创建F值分析结果DataFrame
f_results = pd.DataFrame({
    'Feature': feature_names,
    'F_score': f_scores,
    'p_value': f_pvalues,
    'significant': f_pvalues < 0.05
})

# 按F值排序
f_results = f_results.sort_values('F_score', ascending=False).reset_index(drop=True)

print("F值分析结果（前10个特征）:")
print(f_results.head(10))

# 方法2: 随机森林特征重要性
print("\n=== 方法2: 随机森林特征重要性 ===")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_os, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 获取特征重要性
rf_importance = rf_model.feature_importances_

# 创建随机森林结果DataFrame
rf_results = pd.DataFrame({
    'Feature': feature_names,
    'RF_importance': rf_importance
})

# 按重要性排序
rf_results = rf_results.sort_values('RF_importance', ascending=False).reset_index(drop=True)

print("随机森林特征重要性（前10个特征）:")
print(rf_results.head(10))

# 方法3: LASSO回归
print("\n=== 方法3: LASSO回归 ===")

# 使用LASSO CV选择最佳alpha
lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1)
lasso_cv.fit(X_scaled, y_os)

# 获取LASSO系数
lasso_coef = lasso_cv.coef_

# 创建LASSO结果DataFrame
lasso_results = pd.DataFrame({
    'Feature': feature_names,
    'LASSO_coef': lasso_coef,
    'LASSO_abs_coef': np.abs(lasso_coef)
})

# 按绝对系数排序
lasso_results = lasso_results.sort_values('LASSO_abs_coef', ascending=False).reset_index(drop=True)

print("LASSO回归系数（前10个特征）:")
print(lasso_results.head(10))

# 合并所有结果
print("\n=== 合并所有分析结果 ===")

# 合并三个结果表
final_results = f_results[['Feature', 'F_score', 'p_value']].copy()
final_results = final_results.merge(rf_results[['Feature', 'RF_importance']], on='Feature')
final_results = final_results.merge(lasso_results[['Feature', 'LASSO_coef', 'LASSO_abs_coef']], on='Feature')

# 标准化各方法的分数（0-1范围）
final_results['F_score_norm'] = (final_results['F_score'] - final_results['F_score'].min()) / (final_results['F_score'].max() - final_results['F_score'].min())
final_results['RF_importance_norm'] = (final_results['RF_importance'] - final_results['RF_importance'].min()) / (final_results['RF_importance'].max() - final_results['RF_importance'].min())
final_results['LASSO_coef_norm'] = (final_results['LASSO_abs_coef'] - final_results['LASSO_abs_coef'].min()) / (final_results['LASSO_abs_coef'].max() - final_results['LASSO_abs_coef'].min())

# 计算综合评分（等权重）
final_results['Composite_score'] = (
    final_results['F_score_norm'] * 0.33 +
    final_results['RF_importance_norm'] * 0.33 +
    final_results['LASSO_coef_norm'] * 0.34
)

# 按综合评分排序
final_results = final_results.sort_values('Composite_score', ascending=False).reset_index(drop=True)

print("综合评分结果（前15个特征）:")
print(final_results[['Feature', 'F_score', 'RF_importance', 'LASSO_abs_coef', 'Composite_score']].head(15))

# 保存结果
final_results.to_csv('feature_selection_results.csv', index=False, encoding='utf-8')
f_results.to_csv('f_score_results.csv', index=False, encoding='utf-8')
rf_results.to_csv('rf_importance_results.csv', index=False, encoding='utf-8')
lasso_results.to_csv('lasso_results.csv', index=False, encoding='utf-8')

print(f"\n分析结果已保存:")
print("- feature_selection_results.csv: 综合分析结果")
print("- f_score_results.csv: F值分析结果")
print("- rf_importance_results.csv: 随机森林重要性结果")
print("- lasso_results.csv: LASSO回归结果")

# 输出最重要的前10个特征
print(f"\n=== 最重要的前10个特征 ===")
top_10_features = final_results.head(10)
for i, row in top_10_features.iterrows():
    print(f"{i+1:2d}. {row['Feature']:25s} (综合评分: {row['Composite_score']:.4f})")

print("\n特征选择分析完成！")