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


# 读取数据
df = pd.read_csv('raw_data.csv')

print("=== 数据预处理阶段 ===")
print(f"原始数据形状: {df.shape}")

# 提取特征和标签
# 排除姓名列，OS作为目标变量
feature_columns = [col for col in df.columns if col not in ['姓名', 'OS']]
X = df[feature_columns].copy()
y = df['OS'].copy()

print(f"特征数量: {len(feature_columns)}")
print(f"特征列名: {feature_columns}")

# 检查缺失值
print(f"\n缺失值检查:")
print(f"X中缺失值: {X.isnull().sum().sum()}")
print(f"y中缺失值: {y.isnull().sum()}")

# 检查异常值（使用IQR方法）
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers)

print(f"\n异常值检测（IQR方法）:")
numeric_columns = X.select_dtypes(include=[np.number]).columns
outlier_counts = {}
for col in numeric_columns:
    outlier_counts[col] = detect_outliers_iqr(X, col)
    if outlier_counts[col] > 0:
        print(f"{col}: {outlier_counts[col]} 个异常值")

# 数据标准化
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 创建二元OS变量（OS > 24）
y_binary = (y > 24).astype(int)

print(f"\n数据预处理完成:")
print(f"X_scaled形状: {X_scaled.shape}")
print(f"y形状: {y.shape}")
print(f"y_binary形状: {y_binary.shape}")
print(f"OS > 24的比例: {y_binary.mean():.3f}")

# 保存预处理后的数据
X_scaled.to_csv('X_scaled.csv', index=False)
y.to_csv('y_os.csv', index=False)
y_binary.to_csv('y_binary.csv', index=False)

# 保存特征列名
with open('feature_names.txt', 'w', encoding='utf-8') as f:
    for feature in feature_columns:
        f.write(f"{feature}\n")

print("\n预处理后的数据已保存")
print("- X_scaled.csv: 标准化后的特征矩阵")
print("- y_os.csv: OS值（连续变量）")
print("- y_binary.csv: OS是否大于24（二元变量）")
print("- feature_names.txt: 特征名称列表")