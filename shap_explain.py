import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import shap
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer编码器层"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        # 保存配置，确保与训练脚本一致并可从 h5 安全恢复
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


def preprocess_data_for_shap(df_train, df_test):
    """
    预处理数据，用于SHAP解释。
    """
    # 合并数据集以获取所有可能的特征列
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # 选择特征
    selected_features = [
        'AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
        'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
        'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
        'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
        'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
        'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
        'Cardiopulmonary disease', 'Surgical type', 'Hepatitis'
    ]
    numeric_features = [f for f in selected_features if f in df_combined.columns]

    # 处理训练集
    X_train_df = df_train[numeric_features].astype(float)
    X_train_df_filled = X_train_df.fillna(X_train_df.median())
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df_filled)

    # 处理测试集
    X_test_df = df_test[numeric_features].astype(float)
    X_test_df_filled = X_test_df.fillna(X_train_df.median())  # 使用训练集的中位数
    X_test = scaler.transform(X_test_df_filled)

    return X_train, X_test, X_train_df, X_test_df, numeric_features, scaler


def generate_shap_explanations(model_path, X_train, X_test, X_train_df, X_test_df, feature_names):
    """
    生成SHAP解释图（基于生存模型的线性风险分数）。
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    model = load_model(model_path, custom_objects={'TransformerEncoder': TransformerEncoder}, compile=False)

    # 设置随机种子以保证可复现性
    np.random.seed(40)

    # 随机选择背景数据（安全下限）
    n_background = min(120, X_train.shape[0])
    if n_background == 0:
        raise ValueError("训练集样本数为 0，无法构建 SHAP 背景集。")
    background_indices = np.random.choice(X_train.shape[0], n_background, replace=False)
    background = X_train[background_indices]

    # 随机选择要解释的测试数据（安全下限）
    n_explain = min(50, X_test.shape[0])
    if n_explain == 0:
        raise ValueError("测试集样本数为 0，无法生成 SHAP 解释。")
    test_indices = np.random.choice(X_test.shape[0], n_explain, replace=False)
    test_data = X_test[test_indices]

    # 定义预测函数，返回“风险分数”（线性输出，值越大风险越高）
    def predict_fn(X):
        pred = model.predict(X, verbose=0)
        return pred.flatten()

    # 创建SHAP解释器
    # 使用新的SHAP API
    explainer = shap.Explainer(predict_fn, background)

    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = explainer(test_data)

    # 1. Summary Plot (全局特征重要性)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, test_data, feature_names=feature_names, max_display=15, show=False)
    plt.title('SHAP Summary Plot - Transformer+MLP Survival Risk (Top 15 Features)')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Bar Plot (平均特征重要性)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, test_data, feature_names=feature_names, plot_type="bar", max_display=15, show=False)
    plt.title('SHAP Feature Importance - Transformer+MLP Survival Risk (Top 15)')
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Waterfall Plot (单个样本的解释)
    sample_idx = 0
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], max_display=15, show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {sample_idx} (Risk)')
    plt.tight_layout()
    plt.savefig(f'shap_waterfall_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Force Plot (单个样本的力图)
    force_plot = shap.force_plot(
        shap_values.base_values[sample_idx],
        shap_values.values[sample_idx],
        test_data[sample_idx],
        matplotlib=False
    )
    shap.save_html('shap_force_plot_sample_0.html', force_plot)

    print("SHAP解释图已生成并保存。")


if __name__ == "__main__":
    # 数据文件路径
    train_path = 'train.xlsx'
    test_path = 'test.xlsx'
    model_path = 'best_transformer_mlp_survival_final.h5'

    # 加载数据
    print("加载数据...")
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    # 预处理数据
    X_train, X_test, X_train_df, X_test_df, feature_names, scaler = preprocess_data_for_shap(df_train, df_test)

    # 生成SHAP解释
    generate_shap_explanations(model_path, X_train, X_test, X_train_df, X_test_df, feature_names)
