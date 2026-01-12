import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU, MultiHeadAttention, LayerNormalization, Dropout, Reshape, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import tensorflow as tf
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import joblib

warnings.filterwarnings('ignore')


# --- Custom Layer Definition for Transformer ---
class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer编码器层"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

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


# 数据加载和预处理
def preprocess_split_data(df_train, df_val, df_test):
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    selected_features = ['AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
                             'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
                             'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
                             'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
                             'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
                             'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
                             'Cardiopulmonary disease', 'Surgical type', 'Hepatitis']
    numeric_features = [f for f in selected_features if f in df_combined.columns]

    def process_dataframe(df):
        df['OS_months'] = pd.to_numeric(df['OS'], errors='coerce').clip(0, 180)
        df['event'] = (df['OS_months'] <= 24).astype(int) # Event is death within 24 months
        df['survive_2years'] = (df['OS_months'] > 24).astype(int)
        y_binary = df['survive_2years'].values
        y_surv = Surv.from_arrays(event=df['event'].values, time=df['OS_months'].values)
        
        X_df = df[numeric_features].astype(float)
        return X_df, y_binary, y_surv

    X_train_df, y_binary_train, y_surv_train = process_dataframe(df_train)
    X_val_df, y_binary_val, y_surv_val = process_dataframe(df_val)
    X_test_df, y_binary_test, y_surv_test = process_dataframe(df_test)

    scaler = StandardScaler()
    X_train_df_filled = X_train_df.fillna(X_train_df.median())
    X_train = scaler.fit_transform(X_train_df_filled)
    
    X_val_df_filled = X_val_df.fillna(X_train_df.median())
    X_val = scaler.transform(X_val_df_filled)

    X_test_df_filled = X_test_df.fillna(X_train_df.median())
    X_test = scaler.transform(X_test_df_filled)

    print("数据预处理完成。")
    print(f"训练集: {len(X_train)}例, 验证集: {len(X_val)}例, 测试集: {len(X_test)}例")

    return (X_train, y_binary_train, y_surv_train, X_train_df,
            X_val, y_binary_val, y_surv_val, X_val_df,
            X_test, y_binary_test, y_surv_test, X_test_df,
            numeric_features, scaler)


# --- 新增：单折性能评估函数 ---
def evaluate_fold_metrics(model, fold, X_train, y_train, X_val, y_val, X_test, y_test, y_surv_test, output_file):
    """计算单个折叠的详细性能指标并写入文件"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n--- Fold {fold} Performance Details ---\n")
        
        datasets = {
            "Train": (X_train, y_train),
            "Validation": (X_val, y_val),
            "Test": (X_test, y_test)
        }
        
        results = []
        for name, (X, y) in datasets.items():
            # 处理模型可能在内部做了 Reshape 的情况：根据 model.input_shape 决定是否需要 reshape
            try:
                need_reshape = False
                try:
                    inp_shape = model.input_shape
                    if isinstance(inp_shape, tuple) and len(inp_shape) == 3:
                        need_reshape = True
                except Exception:
                    need_reshape = False

                X_for_pred = np.reshape(X, (X.shape[0], 1, X.shape[1])) if need_reshape else X
                y_pred_proba = model.predict(X_for_pred, verbose=0).ravel()
            except Exception as e:
                f.write(f"预测失败 ({name}): {e}\n")
                # 填充占位以保证后续一致性
                y_pred_proba = np.full((len(y),), np.nan)

            # 如果预测包含 NaN 或全部相同值，跳过部分指标计算
            if np.isnan(y_pred_proba).any() or np.nanstd(y_pred_proba) == 0:
                f.write(f"{name}: 预测无效（包含 NaN 或常数），跳过详细指标计算。\n")
                results.append([name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
                continue

            y_pred_class = (y_pred_proba > 0.5).astype(int)

            # 计算指标，尽量防御性编程，捕获异常并写入 N/A
            try:
                auc = roc_auc_score(y, y_pred_proba)
            except Exception:
                auc = 'N/A'

            try:
                cm = confusion_matrix(y, y_pred_class, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            except Exception:
                accuracy = sensitivity = specificity = ppv = npv = 'N/A'

            try:
                brier = brier_score_loss(y, y_pred_proba)
            except Exception:
                brier = 'N/A'

            # C-index 只能在测试集上计算，因为它需要生存数据
            c_index = 'N/A'
            if name == 'Test':
                try:
                    c_index = concordance_index_censored(y_surv_test['event'], y_surv_test['time'], y_pred_proba)[0]
                except Exception:
                    c_index = 'N/A'

            # 格式化输出
            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) else (str(x) if x is not None else 'N/A')

            results.append([name, fmt(auc), fmt(accuracy), fmt(sensitivity), fmt(specificity), fmt(ppv), fmt(npv), fmt(brier), fmt(c_index)])

        # 准备并写入表格
        headers = ["Dataset", "AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Brier Score", "C-index (Test)"]
        col_widths = [len(h) for h in headers]
        for row in results:
            for i, item in enumerate(row):
                col_widths[i] = max(col_widths[i], len(item))

        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")

        for row in results:
            f.write(" | ".join(item.ljust(w) for item, w in zip(row, col_widths)) + "\n")
        f.write("-" * len(header_line) + "\n")


def compute_classification_metrics(y_true, y_pred_proba):
    """返回基本分类指标字典：AUC, accuracy, sensitivity, specificity, ppv, npv, brier"""
    res = {}
    try:
        res['AUC'] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        res['AUC'] = np.nan
    try:
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        res['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        res['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        res['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        res['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        res['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    except Exception:
        res.update({'Accuracy': np.nan, 'Sensitivity': np.nan, 'Specificity': np.nan, 'PPV': np.nan, 'NPV': np.nan})
    try:
        res['Brier'] = float(brier_score_loss(y_true, y_pred_proba))
    except Exception:
        res['Brier'] = np.nan
    return res


def compute_c_index_ci(y_surv, scores, n_bootstrap=500, seed=42):
    """Bootstrap estimate for c-index 95% CI.
    y_surv: structured array with fields 'event' and 'time'
    scores: array-like risk scores (higher -> higher risk)
    returns: (c_index, ci_low, ci_high)
    """
    try:
        # point estimate
        c_point = float(concordance_index_censored(y_surv['event'], y_surv['time'], scores)[0])
    except Exception:
        return (np.nan, np.nan, np.nan)

    rng = np.random.RandomState(seed)
    n = len(scores)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            c_b = concordance_index_censored(y_surv['event'][idx], y_surv['time'][idx], np.array(scores)[idx])[0]
            boot_vals.append(c_b)
        except Exception:
            continue
    if len(boot_vals) == 0:
        return (c_point, np.nan, np.nan)
    low = np.percentile(boot_vals, 2.5)
    high = np.percentile(boot_vals, 97.5)
    return (c_point, float(low), float(high))


def evaluate_and_print_final_metrics(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, y_surv_test, output_file=None, ci_bootstrap=500):
    """计算并在控制台输出训练/验证/测试集的指标（包含 c-index 和 95% CI）
    model: Keras model
    """
    # 根据 model.input_shape 决定是否需要 reshape
    def prepare_X(X):
        try:
            inp_shape = model.input_shape
            if isinstance(inp_shape, tuple) and len(inp_shape) == 3:
                return np.reshape(X, (X.shape[0], 1, X.shape[1]))
        except Exception:
            pass
        return X

    X_tr = prepare_X(X_train)
    X_va = prepare_X(X_val)
    X_te = prepare_X(X_test)

    y_proba_tr = model.predict(X_tr, verbose=0).ravel()
    y_proba_va = model.predict(X_va, verbose=0).ravel()
    y_proba_te = model.predict(X_te, verbose=0).ravel()

    metrics_tr = compute_classification_metrics(y_train, y_proba_tr)
    metrics_va = compute_classification_metrics(y_val, y_proba_va)
    metrics_te = compute_classification_metrics(y_test, y_proba_te)

    # c-index on test and CI
    c_point, c_low, c_high = compute_c_index_ci(y_surv_test, y_proba_te, n_bootstrap=ci_bootstrap)

    # 打印表格到控制台
    headers = ["Dataset", "AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Brier", "C-index (95% CI)"]
    rows = []
    def fmt(v):
        return f"{v:.4f}" if (isinstance(v, float) and not np.isnan(v)) else (str(v) if v is not None else 'N/A')

    rows.append(["Train", fmt(metrics_tr['AUC']), fmt(metrics_tr['Accuracy']), fmt(metrics_tr['Sensitivity']), fmt(metrics_tr['Specificity']), fmt(metrics_tr['PPV']), fmt(metrics_tr['NPV']), fmt(metrics_tr['Brier']), "-"])
    rows.append(["Validation", fmt(metrics_va['AUC']), fmt(metrics_va['Accuracy']), fmt(metrics_va['Sensitivity']), fmt(metrics_va['Specificity']), fmt(metrics_va['PPV']), fmt(metrics_va['NPV']), fmt(metrics_va['Brier']), "-"])
    ci_str = f"{c_point:.4f} ({c_low:.4f}-{c_high:.4f})" if (not np.isnan(c_point) and not np.isnan(c_low) and not np.isnan(c_high)) else (f"{c_point:.4f}" if not np.isnan(c_point) else 'N/A')
    rows.append(["Test", fmt(metrics_te['AUC']), fmt(metrics_te['Accuracy']), fmt(metrics_te['Sensitivity']), fmt(metrics_te['Specificity']), fmt(metrics_te['PPV']), fmt(metrics_te['NPV']), fmt(metrics_te['Brier']), ci_str])

    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(c)))

    sep = " | "
    print("\n" + "Final performance for model: " + model_name)
    line = sep.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(sep.join(str(c).ljust(w) for c, w in zip(r, col_widths)))
    print("-" * len(line) + "\n")

    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\nFinal performance for model: %s\n" % model_name)
            f.write(line + "\n")
            f.write("-" * len(line) + "\n")
            for r in rows:
                f.write(sep.join(str(c).ljust(w) for c, w in zip(r, col_widths)) + "\n")
            f.write("-" * len(line) + "\n")


# --- 交叉验证和模型训练 ---
def create_transformer_model(input_dim, embed_dim=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, dropout_rate=0.1, use_mlp_head=False):
    """创建Transformer或Transformer+MLP模型"""
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = Dense(embed_dim, activation='relu')(x)
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    
    if use_mlp_head:
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_mlp_model(input_dim):
    """创建一个简单的全连接MLP模型"""
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_sequence_model(input_dim, cell_type='RNN', units=64):
    """创建用于 RNN/LSTM/GRU 的序列模型。输入会被 reshape 为 (1, input_dim)。"""
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    if cell_type == 'RNN':
        x = SimpleRNN(units, activation='tanh')(x)
    elif cell_type == 'LSTM':
        x = LSTM(units)(x)
    elif cell_type == 'GRU':
        x = GRU(units)(x)
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_and_evaluate_cv(X, y, model_name, use_mlp_head, X_test=None, y_test=None, y_surv_test=None, output_file=None):
    """使用5折交叉验证训练和评估模型"""
    plt.rcParams["font.family"] = "Times New Roman"
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 8))

    for i, (train, val) in enumerate(kfold.split(X, y)):
        model = create_transformer_model(X.shape[1], use_mlp_head=use_mlp_head)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
        
        early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=30, restore_best_weights=True, verbose=0)
        
        model.fit(X[train], y[train], epochs=200, batch_size=32,
                  validation_data=(X[val], y[val]),
                  callbacks=[early_stopping], verbose=0)
        
        # 如果是Transformer_MLP，则评估并将详细指标写入文件
        if model_name == 'Transformer_MLP' and X_test is not None and output_file is not None:
            print(f"正在为 Fold {i+1} 生成性能报告...")
            evaluate_fold_metrics(model, i + 1, X[train], y[train], X[val], y[val], X_test, y_test, y_surv_test, output_file)

        y_pred_proba = model.predict(X[val]).ravel()
        fpr, tpr, _ = roc_curve(y[val], y_pred_proba)
        
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = roc_auc_score(y[val], y_pred_proba)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')

        # 保存每折的最佳模型
        model.save(f'best_{model_name}_fold_{i+1}.h5')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})',
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cross-Validation ROC for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'cv_roc_{model_name}.png', dpi=300)
    plt.show()

    # 训练最终模型
    print(f"训练最终的 {model_name} 模型...")
    final_model = create_transformer_model(X.shape[1], use_mlp_head=use_mlp_head)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    final_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=30, restore_best_weights=True, verbose=1)
    
    # 使用一部分验证集进行早停
    from sklearn.model_selection import train_test_split
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    final_model.fit(X_train_final, y_train_final, epochs=200, batch_size=32,
                    validation_data=(X_val_final, y_val_final),
                    callbacks=[early_stopping], verbose=0)
    
    final_model.save(f'best_{model_name}_final.h5')
    print(f"最终 {model_name} 模型已保存为 'best_{model_name}_final.h5'")
    
    return final_model


def train_and_save_cv(model_type, X, y, saved_dir='saved_models', epochs=200, batch_size=32,
                      X_test=None, y_test=None, y_surv_test=None, output_file=None, ci_bootstrap=500):
    """对指定 model_type 执行 5-fold CV，并在每折保存模型文件。

    Keras 模型保存为 .h5，SVM 保存为 .joblib。
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    import joblib

    os.makedirs(saved_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    for train_idx, val_idx in kf.split(X, y):
        fold += 1
        print(f"Training {model_type} - fold {fold} ...")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Special handling for SVM
        if model_type == 'SVM':
            svm = SVC(probability=True, random_state=42)
            svm.fit(X_tr, y_tr)
            save_path = os.path.join(saved_dir, f'best_{model_type}_fold_{fold}.joblib')
            joblib.dump(svm, save_path)
            print(f"Saved SVM fold {fold} to {save_path}")
            continue

        # Create Keras model according to model_type
        if model_type == 'MLP':
            model = create_mlp_model(X.shape[1])
        elif model_type in ['RNN', 'LSTM', 'GRU']:
            model = create_sequence_model(X.shape[1], cell_type=model_type)
        elif model_type == 'Transformer':
            model = create_transformer_model(X.shape[1], use_mlp_head=False)
        elif model_type == 'Transformer_MLP':
            model = create_transformer_model(X.shape[1], use_mlp_head=True)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
        early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=30, restore_best_weights=True, verbose=0)

        X_tr_fit = X_tr
        X_val_fit = X_val

        model.fit(X_tr_fit, y_tr, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val_fit, y_val), callbacks=[early_stopping], verbose=0)

        save_path = os.path.join(saved_dir, f'best_{model_type}_fold_{fold}.h5')
        model.save(save_path)
        print(f"Saved {model_type} fold {fold} to {save_path}")

    # Train final model on full provided data and save
    print(f"Training final {model_type} on full provided training data...")
    if model_type == 'SVM':
        svm_final = SVC(probability=True, random_state=42)
        svm_final.fit(X, y)
        final_path = os.path.join(saved_dir, f'best_{model_type}_final.joblib')
        joblib.dump(svm_final, final_path)
        print(f"Saved final SVM to {final_path}")
        return

    if model_type == 'MLP':
        final_model = create_mlp_model(X.shape[1])
    elif model_type in ['RNN', 'LSTM', 'GRU']:
        final_model = create_sequence_model(X.shape[1], cell_type=model_type)
    elif model_type == 'Transformer':
        final_model = create_transformer_model(X.shape[1], use_mlp_head=False)
    elif model_type == 'Transformer_MLP':
        final_model = create_transformer_model(X.shape[1], use_mlp_head=True)

    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['AUC'])

    X_fit = X

    # use small validation split for early stopping
    from sklearn.model_selection import train_test_split
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_fit, y, test_size=0.1, random_state=42, stratify=y)

    final_model.fit(X_train_final, y_train_final, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val_final, y_val_final), callbacks=[EarlyStopping(monitor='val_auc', mode='max', patience=30, restore_best_weights=True)], verbose=0)

    final_save_path = os.path.join(saved_dir, f'best_{model_type}_final.h5')
    final_model.save(final_save_path)
    print(f"Saved final {model_type} to {final_save_path}")

    # 如果是 Transformer 系列并且提供了测试集，则输出训练/验证/测试的详细指标（包括 c-index 与 95% CI）
    if model_type in ['Transformer', 'Transformer_MLP'] and X_test is not None and y_test is not None and y_surv_test is not None:
        try:
            evaluate_and_print_final_metrics(final_model, model_type,
                                            X_train_final, y_train_final,
                                            X_val_final, y_val_final,
                                            X_test, y_test, y_surv_test,
                                            output_file=output_file,
                                            ci_bootstrap=ci_bootstrap)
        except Exception as e:
            print(f"输出最终指标时出错: {e}")


# 模型加载
def load_all_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        print(f"加载模型: {name} from {path}")
        try:
            if name == 'SVM':
                models[name] = joblib.load(path)
            elif name in ['Transformer', 'Transformer_MLP']:
                 models[name] = load_model(path, custom_objects={'TransformerEncoder': TransformerEncoder}, compile=False)
            else:
                models[name] = load_model(path, compile=False)
        except Exception as e:
            print(f"加载模型 {name} 失败: {e}")
    return models


# 获取预测概率
def get_predictions(models, X_data):
    predictions = {}
    for name, model in models.items():
        try:
            if name == 'SVM':
                pred_proba = model.predict_proba(X_data)[:, 1]
            else:
                # 对于 Keras 模型，检查 model.input_shape 决定是否需要 reshape
                need_reshape = False
                try:
                    inp_shape = model.input_shape
                    # input_shape 可能是 (None, features) 或 (None, 1, features)
                    if isinstance(inp_shape, tuple) and len(inp_shape) == 3:
                        need_reshape = True
                except Exception:
                    # 如果没有 input_shape 属性或读取失败，默认不 reshape
                    need_reshape = False

                if need_reshape:
                    X_for_pred = np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))
                else:
                    X_for_pred = X_data

                pred = model.predict(X_for_pred, verbose=0)
                pred_proba = pred.flatten()
        except Exception as e:
            print(f"预测时模型 {name} 出错: {e}")
            # 填充 NaN 以保持字典键一致性
            pred_proba = np.full((X_data.shape[0],), np.nan)
        predictions[name] = pred_proba
    return predictions


# 绘制ROC曲线
def plot_roc_curves(predictions, y_true, save_name="roc_comparison"):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 8))

    for name, pred_proba in predictions.items():
        # 跳过包含 NaN 或常数预测的模型
        if pred_proba is None or np.isnan(pred_proba).any():
            print(f"跳过 {name}：预测包含 NaN。")
            continue
        try:
            if np.nanstd(pred_proba) == 0:
                print(f"跳过 {name}：预测为常数，无法绘制 ROC。")
                continue
        except Exception:
            pass

        try:
            fpr, tpr, _ = roc_curve(y_true, pred_proba)
            auc = roc_auc_score(y_true, pred_proba)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (Test, AUC={auc:.4f})')
        except Exception as e:
            print(f"绘制 {name} ROC 时出错: {e}")
            continue

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = f'{save_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC图像已保存为: {save_path}")
    plt.show()


# --- 新增：KM曲线绘制功能 ---
def plot_km_curves(predictions, y_surv, save_name_prefix="km_curve"):
    """为每个模型绘制KM生存曲线"""
    plt.rcParams["font.family"] = "Times New Roman"
    for name, pred_proba in predictions.items():
        plt.figure(figsize=(8, 6))
        # 跳过无效预测
        if pred_proba is None or np.isnan(pred_proba).any():
            print(f"跳过 {name} 的 KM 曲线：预测包含 NaN。")
            continue

        # 根据预测概率中位数将患者分为高风险和低风险组
        try:
            median_risk = np.median(pred_proba)
        except Exception as e:
            print(f"无法计算 {name} 的中位数风险: {e}")
            continue
        high_risk_mask = pred_proba >= median_risk
        low_risk_mask = pred_proba < median_risk
        
        T = y_surv['time']
        E = y_surv['event']

        # 确保每组至少有一个样本
        if sum(high_risk_mask) == 0 or sum(low_risk_mask) == 0:
            print(f"跳过 {name} 的 KM 曲线：某一组样本为空（high={sum(high_risk_mask)}, low={sum(low_risk_mask)})。")
            continue

        kmf_high = KaplanMeierFitter()
        kmf_high.fit(T[high_risk_mask], event_observed=E[high_risk_mask], label=f'High Risk (n={sum(high_risk_mask)})')
        ax = kmf_high.plot_survival_function()

        kmf_low = KaplanMeierFitter()
        kmf_low.fit(T[low_risk_mask], event_observed=E[low_risk_mask], label=f'Low Risk (n={sum(low_risk_mask)})')
        kmf_low.plot_survival_function(ax=ax)
        
        # Log-rank检验
        results = logrank_test(T[high_risk_mask], T[low_risk_mask], E[high_risk_mask], E[low_risk_mask])
        p_value = results.p_value

        plt.title(f'Kaplan-Meier Curve for {name}\n(Log-rank p-value: {p_value:.4f})')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        
        save_path = f'{save_name_prefix}_{name}.png'
        plt.savefig(save_path, dpi=300)
        print(f"KM曲线已保存为: {save_path}")
        plt.show()


# 主程序
if __name__ == "__main__":
    import os
    import argparse
    train_path = 'train.xlsx'
    val_path = 'val.xlsx'
    test_path = 'test.xlsx'
    output_file_path = 'D:\\xjtu\\fishbook\\output.txt'

    # 每次运行时清空输出文件
    if os.path.exists(output_file_path):
        open(output_file_path, 'w').close()
        print(f"'{output_file_path}' 已清空。")

    print("=== 加载并预处理数据集 ===")
    try:
        df_train = pd.read_excel(train_path)
        df_val = pd.read_excel(val_path)
        df_test = pd.read_excel(test_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}")
        exit()

    (X_train, y_binary_train, y_surv_train, X_train_df,
     X_val, y_binary_val, y_surv_val, X_val_df,
     X_test, y_binary_test, y_surv_test, X_test_df,
     numeric_features, scaler) = preprocess_split_data(df_train, df_val, df_test)

    # 解析命令行参数：允许用户指定只训练哪些模型（逗号分隔）
    parser = argparse.ArgumentParser(description='Train selected models and save per-fold models.')
    parser.add_argument('--models', type=str, default='all', help="Comma-separated model names to train. Choices: Transformer_MLP,Transformer,MLP,RNN,LSTM,GRU,SVM or 'all' (default). Case-insensitive.")
    args = parser.parse_args()

    # 规范化用户输入为标准名称集合
    choice_raw = [s.strip() for s in args.models.split(',') if s.strip()]
    mapping = {
        'mlp': 'MLP', 'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU',
        'transformer': 'Transformer', 'transformer_mlp': 'Transformer_MLP', 'svm': 'SVM', 'all': 'ALL'
    }
    chosen = set()
    for item in choice_raw:
        key = item.lower()
        if key in mapping:
            chosen.add(mapping[key])
        else:
            print(f"未知模型名: {item}，已忽略。可用名称: {', '.join(mapping.keys())}")

    all_models = ['Transformer_MLP', 'Transformer', 'MLP', 'RNN', 'LSTM', 'GRU', 'SVM']
    if 'ALL' in chosen or not chosen:
        requested_models = all_models
    else:
        # 保持 order in all_models
        requested_models = [m for m in all_models if m in chosen]

    print(f"将训练的模型: {requested_models}")

    # 现在对各个被请求的模型执行训练
    saved_models_dir = 'saved_models'
    os.makedirs(saved_models_dir, exist_ok=True)

    # 基础训练集
    X_base = np.concatenate([X_train, X_val], axis=0)
    y_base = np.concatenate([y_binary_train, y_binary_val], axis=0)

    # 条件执行 Transformer_MLP
    if 'Transformer_MLP' in requested_models:
        print("\n=== 训练 Transformer+MLP (5-fold CV 并保存每折) ===")
        train_and_save_cv('Transformer_MLP', X_base, y_base, saved_dir=saved_models_dir,
                          X_test=X_test, y_test=y_binary_test, y_surv_test=y_surv_test, output_file=output_file_path, ci_bootstrap=500)

    # 条件执行 Transformer
    if 'Transformer' in requested_models:
        print("\n=== 训练 Transformer (5-fold CV 并保存每折) ===")
        train_and_save_cv('Transformer', X_base, y_base, saved_dir=saved_models_dir,
                          X_test=X_test, y_test=y_binary_test, y_surv_test=y_surv_test, output_file=output_file_path, ci_bootstrap=500)

    # 其他模型（使用基础训练集）
    for m in ['MLP', 'RNN', 'LSTM', 'GRU', 'SVM']:
        if m in requested_models:
            print(f"\n=== 训练并保存 {m} 的 5-fold 模型 ===")
            train_and_save_cv(m, X_base, y_base, saved_dir=saved_models_dir)

    # --- 加载所有模型 ---
    print("\n=== 加载所有模型进行比较 ===")
    model_paths = {
        'SVM': 'saved_models\\best_SVM_final.joblib', # 假设SVM模型已保存
        'MLP': 'saved_models\\best_MLP_final.h5',
        'RNN': 'saved_models\\best_RNN_final.h5',
        'LSTM': 'saved_models\\best_LSTM_final.h5',
        'GRU': 'saved_models\\best_GRU_final.h5',
        'Transformer': 'saved_models\\best_Transformer_final.h5',
        'Transformer_MLP': 'saved_models\\best_Transformer_MLP_final.h5'
    }

    if not os.path.exists(model_paths['SVM']):
        print("训练并保存一个SVM模型...")
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_train, y_binary_train)
        joblib.dump(svm_model, model_paths['SVM'])
    else:
        print(f"SVM 文件已存在 ({model_paths['SVM']})，跳过训练。")
    
    models = load_all_models(model_paths)

    print("\n=== 在测试集上生成预测结果 ===")
    test_predictions = get_predictions(models, X_test)

    print("\n=== 在测试集上绘制ROC曲线 ===")
    plot_roc_curves(test_predictions, y_binary_test, save_name="roc_comparison_test_set")

    # print("\n=== 在测试集上绘制KM生存曲线 ===")
    # plot_km_curves(test_predictions, y_surv_test, save_name_prefix="km_curve_test_set")
