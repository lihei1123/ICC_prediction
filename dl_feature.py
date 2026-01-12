import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, GlobalAveragePooling1D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# === TransformerEncoder copied/adapted from 4.py ===
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = "Times New Roman"

class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        from tensorflow.keras import Sequential
        self.ffn = Sequential([Dense(ff_dim, activation='relu'), Dense(embed_dim)])
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
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config


# 全局按重要性排序的特征顺序
SELECTED_FEATURES_ORDERED = ['AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
                             'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
                             'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
                             'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
                             'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
                             'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
                             'Cardiopulmonary disease', 'Surgical type', 'Hepatitis']


def preprocess_split_data(df_train, df_val, df_test, selected_features=None):
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)
    if selected_features is None:
        selected_features = SELECTED_FEATURES_ORDERED
    numeric_features = [f for f in selected_features if f in df_combined.columns]

    def process_dataframe(df):
        df['OS_months'] = pd.to_numeric(df['OS'], errors='coerce').clip(0, 180)
        df['survive_2years'] = (df['OS_months'] > 24).astype(int)
        y_binary = df['survive_2years'].values
        # return dataframe of numeric features (unscaled) for flexible scaling later
        X_df = df[numeric_features].astype(float)
        return X_df, y_binary

    X_train_df, y_binary_train = process_dataframe(df_train)
    X_val_df, y_binary_val = process_dataframe(df_val)
    X_test_df, y_binary_test = process_dataframe(df_test)

    print('数据预处理完成。')
    print(f'训练集: {len(X_train_df)}例, 验证集: {len(X_val_df)}例, 测试集: {len(X_test_df)}例')

    return (X_train_df, y_binary_train,
            X_val_df, y_binary_val,
            X_test_df, y_binary_test,
            numeric_features)


# ---------- Model builders (MLP, RNN, LSTM, GRU, Transformer, Transformer+MLP) ----------

def create_mlp_model(input_dim, hidden_units=(64, 32), dropout_rate=0.2):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for h in hidden_units:
        x = Dense(h, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_rnn_model(input_dim, rnn_units=50, dropout_rate=0.0):
    """
    - units=50, activation='relu'
    - 不使用 RNN dropout（与导出的模型一致），但保留可配置的 dropout_rate（默认0.0）以便实验
    """
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    # best_rnn_model 使用 units=50, activation='relu'
    x = SimpleRNN(rnn_units, activation='relu')(x)
    if dropout_rate and dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_lstm_model(input_dim, lstm_units=50, dropout_rate=0.0):
    """
    - units=50, activation='relu', recurrent_activation='sigmoid'
    - implementation=2 保持与导出模型兼容
    """
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = LSTM(lstm_units, activation='relu', recurrent_activation='sigmoid', implementation=2)(x)
    if dropout_rate and dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_gru_model(input_dim, gru_units=50, dropout_rate=0.0):
    """
    - units=50, activation='relu', recurrent_activation='sigmoid'
    - reset_after=True, implementation=2 与导出模型保持一致
    """
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = GRU(gru_units, activation='relu', recurrent_activation='sigmoid', reset_after=True, implementation=2)(x)
    if dropout_rate and dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_transformer_model(input_dim, embed_dim=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, use_mlp_head=False, dropout_rate=0.1):
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = Dense(embed_dim, activation='relu')(x)
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, rate=dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    if use_mlp_head:
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# ---------- Training helpers ----------

def compile_and_fit(model, X_train, y_train, X_val=None, y_val=None, epochs=60, batch_size=32, patience=10, verbose=0):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['AUC'])
    callbacks = []
    if (X_val is not None) and (y_val is not None):
        early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=patience, restore_best_weights=True, verbose=0)
        callbacks.append(early_stopping)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    return history


def evaluate_model_on_test(model, X_test, y_test):
    # model.predict returns probabilities
    pred = model.predict(X_test, verbose=0).ravel()
    try:
        auc = roc_auc_score(y_test, pred)
    except Exception:
        auc = np.nan
    return auc, pred


# ---------- Feature selection experiment for deep models ----------

def feature_selection_experiment_deep(selected_ordered_features,
                                      X_train_df, y_train,
                                      X_val_df, y_val,
                                      X_test_df, y_test,
                                      model_types=('Transformer', 'Transformer_MLP', 'MLP', 'RNN', 'LSTM', 'GRU'),
                                      max_k=None,
                                      epochs=60,
                                      batch_size=32,
                                      use_cv=True,
                                      n_splits=5,
                                      n_repeats=1,
                                      patience=10,
                                      save_csv='feature_selection_results_deep.csv',
                                      save_models=True,
                                      model_save_dir='saved_models_topk_deep2'):
    """
    对深度模型按 top-k 特征做实验。
    - 默认 use_cv=False（对深度模型逐 k 单次训练更快）；若 use_cv=True，会用 StratifiedKFold 对训练集进行 n_splits 折内验证并返回平均验证 AUC，之后在 train+val 上训练最终模型并测试。
    - model_types: 要评估的模型类型（字符串列表）。
    返回 df 记录每个 k, model 的 test AUC（与可选的 cv_val_auc）
    """
    results = []
    if max_k is None:
        max_k = len(selected_ordered_features)
    max_k = min(max_k, len(selected_ordered_features))

    print(f"开始深度模型 top-k 实验，k 从 1 到 {max_k}，models={model_types}, use_cv={use_cv}")

    for k in range(1, max_k + 1):
        selected = [f for f in selected_ordered_features[:k] if f in X_train_df.columns]
        if len(selected) == 0:
            print(f"k={k} 时没有可用特征，跳过")
            continue

        t0 = time.time()
        # 填充训练中位数并标准化
        train_subset = X_train_df[selected].copy()
        val_subset = X_val_df[selected].copy()
        test_subset = X_test_df[selected].copy()

        train_median = train_subset.median()
        X_train_filled = train_subset.fillna(train_median).values
        X_val_filled = val_subset.fillna(train_median).values
        X_test_filled = test_subset.fillna(train_median).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        X_test_scaled = scaler.transform(X_test_filled)

        # 合并 train+val 的基础训练集
        X_train_full_base = np.concatenate([X_train_scaled, X_val_scaled], axis=0)
        y_train_full_base = np.concatenate([y_train, y_val], axis=0)

        for mtype in model_types:
            cv_val_auc = np.nan
            # 每个模型都从基准 train+val 开始
            X_train_full = X_train_full_base.copy()
            y_train_full = y_train_full_base.copy()
            
            if use_cv:
                # 做简单的折内验证以评估模型稳定性
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                val_aucs = []
                for train_idx, val_idx in skf.split(X_train_full, y_train_full):
                    X_tr, X_va = X_train_full[train_idx], X_train_full[val_idx]
                    y_tr, y_va = y_train_full[train_idx], y_train_full[val_idx]

                    K.clear_session()
                    if mtype == 'Transformer':
                        model = create_transformer_model(X_tr.shape[1], use_mlp_head=False)
                    elif mtype == 'Transformer_MLP':
                        model = create_transformer_model(X_tr.shape[1], use_mlp_head=True)
                    elif mtype == 'MLP':
                        model = create_mlp_model(X_tr.shape[1])
                    elif mtype == 'RNN':
                        model = create_rnn_model(X_tr.shape[1])
                    elif mtype == 'LSTM':
                        model = create_lstm_model(X_tr.shape[1])
                    elif mtype == 'GRU':
                        model = create_gru_model(X_tr.shape[1])
                    else:
                        raise ValueError(f'未知模型类型: {mtype}')

                    compile_and_fit(model, X_tr, y_tr, X_val=X_va, y_val=y_va, epochs=epochs, batch_size=batch_size, patience=patience, verbose=0)
                    auc_val, _ = evaluate_model_on_test(model, X_va, y_va)
                    val_aucs.append(auc_val)
                cv_val_auc = np.nanmean(val_aucs)

            # 最终在 train+val 上训练并评估测试集
            # 如果 n_repeats>1，则对每个 (k, model) 重复训练 n_repeats 次并统计 mean/std
            from sklearn.model_selection import train_test_split
            auc_tests = []
            for rep in range(max(1, n_repeats)):
                K.clear_session()
                if mtype == 'Transformer':
                    model_final = create_transformer_model(X_train_full.shape[1], use_mlp_head=False)
                elif mtype == 'Transformer_MLP':
                    model_final = create_transformer_model(X_train_full.shape[1], use_mlp_head=True)
                elif mtype == 'MLP':
                    model_final = create_mlp_model(X_train_full.shape[1])
                elif mtype == 'RNN':
                    model_final = create_rnn_model(X_train_full.shape[1])
                elif mtype == 'LSTM':
                    model_final = create_lstm_model(X_train_full.shape[1])
                elif mtype == 'GRU':
                    model_final = create_gru_model(X_train_full.shape[1])
                else:
                    raise ValueError(f'未知模型类型: {mtype}')

                # 使用一小部分验证用于早停（从 train_full 中划出 10%），每次重复尝试不同的 random_state
                try:
                    X_trn, X_val_hold, y_trn, y_val_hold = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42 + rep, stratify=y_train_full)
                except Exception:
                    X_trn, X_val_hold, y_trn, y_val_hold = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42 + rep)

                compile_and_fit(model_final, X_trn, y_trn, X_val=X_val_hold, y_val=y_val_hold, epochs=epochs, batch_size=batch_size, patience=patience, verbose=0)
                auc_test_rep, preds_test = evaluate_model_on_test(model_final, X_test_scaled, y_test)
                auc_tests.append(auc_test_rep)

            auc_test_mean = np.nanmean(auc_tests) if len(auc_tests) > 0 else np.nan
            auc_test_std = np.nanstd(auc_tests) if len(auc_tests) > 0 else np.nan

            results.append({
                'k': k,
                'n_features': len(selected),
                'model': mtype,
                'cv_val_auc': cv_val_auc,
                'auc_test': auc_test_mean,
                'auc_test_std': auc_test_std
            })
            if n_repeats > 1:
                print(f"k={k:2d} | model={mtype:15s} | test AUC={auc_test_mean:.4f} (std={auc_test_std:.4f}, n={len(auc_tests)}) | cv_val_auc={cv_val_auc if not np.isnan(cv_val_auc) else 'N/A'}")
            else:
                print(f"k={k:2d} | model={mtype:15s} | test AUC={auc_test_mean:.4f} | cv_val_auc={cv_val_auc if not np.isnan(cv_val_auc) else 'N/A'}")

        elapsed = time.time() - t0
        print(f"k={k} 完成，耗时 {elapsed:.1f}s\n")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res.to_csv(save_csv, index=False)
        print(f"实验结果已保存为: {os.path.abspath(save_csv)}")
        try:
            summary = df_res.groupby('k')['auc_test'].agg(['mean', 'max']).reset_index()
            plt.rcParams["font.family"] = "Times New Roman"
            plt.figure(figsize=(10, 8))
            plt.plot(summary['k'], summary['mean'], marker='o', lw=2, linestyle='-', label='Mean test AUC (across models)')
            plt.plot(summary['k'], summary['max'], marker='s', lw=2, linestyle='-', label='Best test AUC (across models)')
            plt.xlabel('Number of top features (k)', fontsize=12)
            plt.ylabel('AUC on test set', fontsize=12)
            plt.title('Feature selection (top-k) — Deep models', fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend(loc='lower right', fontsize=10)
            plt.tight_layout()
            png = save_csv.replace('.csv', '3.png')
            plt.savefig(png, dpi=300, bbox_inches='tight')
            print(f"汇总图片已保存为: {os.path.abspath(png)}")
            plt.show()
        except Exception as e:
            print('绘图失败:', e)
        # --- 生成每个模型单独的 k vs AUC 曲线（与 3.py 风格一致）
        try:
            pivot = df_res.pivot(index='k', columns='model', values='auc_test')
            plt.rcParams["font.family"] = "Times New Roman"
            plt.figure(figsize=(14, 8))
            # x 轴使用 1..N 的刻度（对应前 k 个特征）
            x = list(range(1, len(pivot) + 1))
            for col in pivot.columns:
                plt.plot(x, pivot[col].values, marker='o', linestyle='-', lw=2, label=col)

            plt.xlabel('Number of Features', fontsize=12)
            plt.ylabel('Test Set AUC', fontsize=12)
            plt.title('Model Performance vs. Number of Features', fontsize=14)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(loc='best', fontsize=10)
            plt.xticks(range(1, len(pivot) + 1))
            # 纵轴自动缩放（不强制从 0 开始）
            plt.tight_layout()
            per_model_png = save_csv.replace('.csv', '_per_model3.png')
            plt.savefig(per_model_png, dpi=300, bbox_inches='tight')
            print(f"每模型随 k 曲线已保存为: {os.path.abspath(per_model_png)}")
            plt.show()
        except Exception as e:
            print('每模型绘图失败:', e)
        # --- 保存模型: 保存每个模型在其最佳 k 时的 h5，以及 k=31 时的 h5 ---
        if save_models:
            # 确保在扫描时已启用 CV
            if not use_cv:
                print('注意: 要求保存模型，已自动将 use_cv 设置为 True 以开启交叉验证用于更稳定的选择。')
                use_cv = True

            os.makedirs(model_save_dir, exist_ok=True)

            def _train_and_save_model_for(mtype, k, out_path):
                # 为指定模型和 k 重新准备数据并训练最终模型，然后保存为 h5
                selected = [f for f in selected_ordered_features[:k] if f in X_train_df.columns]
                if len(selected) == 0:
                    print(f"无法为模型 {mtype} 保存 k={k} 的模型：没有可用特征，跳过。")
                    return False

                # 填充与标准化
                train_subset = X_train_df[selected].copy()
                val_subset = X_val_df[selected].copy()
                test_subset = X_test_df[selected].copy()

                train_median = train_subset.median()
                X_train_filled = train_subset.fillna(train_median).values
                X_val_filled = val_subset.fillna(train_median).values
                X_test_filled = test_subset.fillna(train_median).values

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_filled)
                X_val_scaled = scaler.transform(X_val_filled)
                X_test_scaled = scaler.transform(X_test_filled)

                X_train_full = np.concatenate([X_train_scaled, X_val_scaled], axis=0)
                y_train_full = np.concatenate([y_train, y_val], axis=0)

                # 最终训练并保存
                K.clear_session()
                if mtype == 'Transformer':
                    model_final = create_transformer_model(X_train_full.shape[1], use_mlp_head=False)
                elif mtype == 'Transformer_MLP':
                    model_final = create_transformer_model(X_train_full.shape[1], use_mlp_head=True)
                elif mtype == 'MLP':
                    model_final = create_mlp_model(X_train_full.shape[1])
                elif mtype == 'RNN':
                    model_final = create_rnn_model(X_train_full.shape[1])
                elif mtype == 'LSTM':
                    model_final = create_lstm_model(X_train_full.shape[1])
                elif mtype == 'GRU':
                    model_final = create_gru_model(X_train_full.shape[1])
                else:
                    print(f"未知模型类型: {mtype}，跳过保存")
                    return False

                # 用 10% 的 hold-out 做早停
                try:
                    X_trn, X_val_hold, y_trn, y_val_hold = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)
                except Exception:
                    X_trn, X_val_hold, y_trn, y_val_hold = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

                compile_and_fit(model_final, X_trn, y_trn, X_val=X_val_hold, y_val=y_val_hold, epochs=epochs, batch_size=batch_size, patience=patience, verbose=0)
                try:
                    model_final.save(out_path)
                    print(f"已保存模型: {out_path}")
                    return True
                except Exception as e:
                    print(f"保存模型失败 ({out_path}):", e)
                    return False

            # 遍历每个模型，找出其在扫描中表现最好的 k 并保存对应模型，以及保存 k=31 的模型
            models = df_res['model'].unique()
            for m in models:
                # 找到 test AUC 最好的行
                try:
                    best_row = df_res[df_res['model'] == m].sort_values('auc_test', ascending=False).iloc[0]
                except Exception:
                    print(f"在结果中未找到模型 {m} 的记录，跳过保存。")
                    continue
                best_k = int(best_row['k'])
                fname_best = os.path.join(model_save_dir, f"{m}_best_k{best_k}.h5")
                _train_and_save_model_for(m, best_k, fname_best)

                # 保存 k=31 的模型
                k31 = min(max_k, 31)
                fname_k31 = os.path.join(model_save_dir, f"{m}_k{k31}.h5")
                _train_and_save_model_for(m, k31, fname_k31)
        # --- 在控制台打印汇总 ---
        try:
            print('\n=== 特征选择实验结果 ===')
            # 每个 k 的均值和最大值
            summary = df_res.groupby('k').agg({'auc_test': ['mean', 'max'], 'n_features': 'first'})
            summary.columns = ['mean_auc', 'max_auc', 'n_features']
            summary = summary.reset_index()
            # 打印前 20 行
            print('\nk  n_features  mean_auc  max_auc')
            for _, row in summary.head(20).iterrows():
                print(f"{int(row['k']):2d}  {int(row['n_features']):3d}        {row['mean_auc']:.4f}   {row['max_auc']:.4f}")

            print('\n--- 各模型最佳表现 ---')
            # 每个模型在所有 k 中的最佳表现
            best_per_model = df_res.loc[df_res.groupby('model')['auc_test'].idxmax()].copy()
            best_per_model = best_per_model.sort_values(by='auc_test', ascending=False)
            print('\nmodel               best_k  n_features  best_auc   cv_val_auc')
            for _, r in best_per_model.iterrows():
                cv_val = f"{r['cv_val_auc']:.4f}" if not pd.isna(r['cv_val_auc']) else 'N/A'
                print(f"{r['model']:18s}  {int(r['k']):6d}  {int(r['n_features']):9d}  {r['auc_test']:.4f}  {cv_val}")

            # 全局最佳
            try:
                best_global = df_res.loc[df_res['auc_test'].idxmax()]
                print('\n--- 全局最佳 ---')
                print(f"Model={best_global['model']}, k={int(best_global['k'])}, n_features={int(best_global['n_features'])}, auc_test={best_global['auc_test']:.4f}")
            except Exception:
                pass
        except Exception as e:
            print('生成控制台汇总失败:', e)
    else:
        print('没有实验结果。')

    return df_res


# ---------- Main: loading data and optional experiment ----------
if __name__ == '__main__':
    train_path = 'train.xlsx'
    val_path = 'val.xlsx'
    test_path = 'test.xlsx'

    try:
        df_train = pd.read_excel(train_path)
        df_val = pd.read_excel(val_path)
        df_test = pd.read_excel(test_path)
        print('成功加载数据文件。')
    except FileNotFoundError as e:
        print('找不到数据文件:', e.filename)
        raise

    (X_train_df, y_train,
     X_val_df, y_val,
     X_test_df, y_test,
     numeric_features) = preprocess_split_data(df_train, df_val, df_test)

    # 实验开关：现在默认开启以便直接运行 top-k 重复实验（每个 k 重复 n_repeats 次以取均值与 std）
    run_feature_experiment = True

    if run_feature_experiment:
        # 实验设置：每个 k 在每个模型上重复训练多次以估计 test AUC 的均值与标准差（n_repeats 可调整）
        df_exp = feature_selection_experiment_deep(SELECTED_FEATURES_ORDERED,
                                                  X_train_df, y_train,
                                                  X_val_df, y_val,
                                                  X_test_df, y_test,
                                                  model_types=('Transformer','Transformer_MLP','MLP','RNN','LSTM','GRU'),
                                                  max_k=31,
                                                  epochs=60,
                                                  batch_size=32,
                                                  use_cv=True,
                                                  n_splits=5,
                                                  n_repeats=5,
                                                  patience=10,
                                                  save_csv='feature_selection_results_deep.csv')
        if not df_exp.empty:
            print('\nTop-k 深度模型实验完成，摘要（前 10 行）:')
            print(df_exp.groupby('k')['auc_test'].agg(['mean','max']).head(10))

    else:
        print('run_feature_experiment 默认关闭。如需运行请将 run_feature_experiment 设为 True 并调整参数（epochs, max_k, use_cv 等）。')

    print('脚本结束。')
