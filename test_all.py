import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')


# --- 新增函数：评估并打印模型性能 ---
def _bootstrap_cindex(risk_scores, y_surv, n_boot=1000, random_state=43):
    """使用自助法(bootstrap)基于 sksurv 的 concordance_index_censored 估计C-index的95%置信区间。
    risk_scores: 一维风险分数数组（数值越大代表风险越高，与 wuzhe.py 一致）
    y_surv: dict-like，包含 'time' 与 'event' 字段（event需为bool或可转为bool）
    返回 (c_index, ci_lower, ci_upper)
    """
    from sksurv.metrics import concordance_index_censored
    rng = np.random.RandomState(random_state)
    times = np.asarray(y_surv['time'])
    events = np.asarray(y_surv['event']).astype(bool)
    preds = np.asarray(risk_scores).flatten()

    c_orig = concordance_index_censored(events, times, preds)[0]

    # Bootstrap 计算置信区间
    n = len(preds)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        c = concordance_index_censored(events[idx], times[idx], preds[idx])[0]
        boot_scores.append(c)

    lower = float(np.percentile(boot_scores, 2.5))
    upper = float(np.percentile(boot_scores, 97.5))
    return float(c_orig), lower, upper


def compute_metrics_for_model(pred_proba, y_true):
    """计算二分类相关指标并返回字典。
    pred_proba: 1d array
    y_true: 1d binary array
    返回: dict 包含 AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, Brier
    """
    out = {}
    if y_true is None or len(np.unique(y_true)) < 2:
        return None

    # AUC
    try:
        out['AUC'] = float(roc_auc_score(y_true, pred_proba))
    except Exception:
        out['AUC'] = np.nan

    # Threshold 0.5分类
    y_pred = (np.asarray(pred_proba) >= 0.5).astype(int)

    out['Accuracy'] = float(accuracy_score(y_true, y_pred))
    out['Sensitivity'] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out['Specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    out['PPV'] = float(precision_score(y_true, y_pred, zero_division=0))
    out['NPV'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # Brier Score
    try:
        out['Brier'] = float(brier_score_loss(y_true, pred_proba))
    except Exception:
        out['Brier'] = np.nan

    return out


def evaluate_and_print_metrics(predictions, y_true, dataset_name, y_surv=None, compute_ci=False, n_boot=1000):
    """计算并打印一组模型在某个数据集上的指标。
    如果 compute_ci=True 且 y_surv 提供，则对 C-index 计算95% CI（bootstrap）。
    """
    print(f"\n--- {dataset_name} 评估结果 ---")
    headers = ['Model', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Brier']
    print(f"{headers[0]:<25} | {headers[1]:<6} | {headers[2]:<8} | {headers[3]:<10} | {headers[4]:<11} | {headers[5]:<6} | {headers[6]:<6} | {headers[7]:<7}")
    print("-" * 110)

    for name, pred_proba in predictions.items():
        metrics = compute_metrics_for_model(pred_proba, y_true)
        if metrics is None:
            print(f"{name:<25} | {'N/A (insufficient labels)':<85}")
            continue

        # 格式化基础指标
        print(f"{name:<25} | {metrics['AUC']:<6.4f} | {metrics['Accuracy']:<8.4f} | {metrics['Sensitivity']:<10.4f} | {metrics['Specificity']:<11.4f} | {metrics['PPV']:<6.4f} | {metrics['NPV']:<6.4f} | {metrics['Brier']:<7.4f}")

    # 如果需要，计算并打印 C-index（尤其是测试集）和95% CI
    if y_surv is not None:
        # 与 wuzhe.py 保持一致：使用 sksurv.metrics.concordance_index_censored，风险分数越大风险越高
        from sksurv.metrics import concordance_index_censored
        print("\nC-index on this dataset:")
        print(f"{'Model':<25} | {'C-index':<8} | {'95% CI':<20}")
        print("-" * 60)
        for name, pred_values in predictions.items():
            try:
                times = np.asarray(y_surv['time'])
                events = np.asarray(y_surv['event']).astype(bool)

                # 风险分数定义：
                # - 二分类输出概率 -> 风险 = 1 - p
                # - 直接输出风险的模型 -> 直接使用（CoxPH, RSF）。
                #   注意：Transformer_MLP 在 predictions 中已转换为伪概率，这里按概率处理（risk = 1 - proba）。
                if name in ['CoxPH', 'Random Survival Forest']:
                    risk_scores = np.asarray(pred_values).flatten()
                else:
                    risk_scores = 1 - np.asarray(pred_values).flatten()

                c = concordance_index_censored(events, times, risk_scores)[0]

                if compute_ci:
                    try:
                        c_orig, lower, upper = _bootstrap_cindex(risk_scores, y_surv, n_boot=n_boot)
                        print(f"{name:<25} | {c_orig:<8.4f} | ({lower:.4f}, {upper:.4f})")
                    except Exception as e:
                        print(f"{name:<25} | {c:<8.4f} | CI计算失败: {e}")
                else:
                    print(f"{name:<25} | {c:<8.4f} | {'N/A':<20}")
            except Exception as e:
                print(f"{name:<25} | Error computing C-index: {e}")


# --- Custom Layer Definition for Transformer ---
class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer编码器层"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
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


# 复用原数据加载和预处理函数
def preprocess_split_data(df_train, df_val, df_test):
    """
    对已经划分好的训练、验证、测试集进行预处理。
    Scaler在训练集上拟合，并应用于所有数据集。
    """
    # 合并数据集以获取所有可能的特征列
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # 选择特征
    selected_features = ['AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
                             'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
                             'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
                             'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
                             'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
                             'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
                             'Cardiopulmonary disease', 'Surgical type', 'Hepatitis']
    # selected_features = ['Age', 'BMI',  'Hepatocirrhosis',
    #                      'Lymph node dissection', 'Peroperative bleeding',
    #                      'AC', 'PLR', 'NLR', 'ALT', 'GGT', 'TBIL', 
    #                      'CEA', 'CA199', 'Tumor differentiation',
    #                      'Tumor number', 'Tumor size', 'Lymphatic metastasis',
    #                      ]
    numeric_features = [f for f in selected_features if f in df_combined.columns]

    # 定义一个内部函数来处理单个DataFrame
    def process_dataframe(df):
        df['OS_months'] = pd.to_numeric(df['OS'], errors='coerce').clip(0, 180)
        # 二分类标签：是否存活超过2年 (1=超过2年仍存活或随访，0=2年内死亡)
        df['survive_2years'] = (df['OS_months'] > 24).astype(int)
        y_binary = df['survive_2years'].values
        # 生存分析事件标签：是否在2年内死亡 (1=死亡事件发生，0=截尾/未在2年内死亡)
        death_within_2y = (df['OS_months'] <= 24).astype(int)
        y_surv = Surv.from_arrays(event=death_within_2y.values, time=df['OS_months'].values)
        
        X_df = df[numeric_features].astype(float)
        return X_df, y_binary, y_surv

    # 分别处理每个数据集
    X_train_df, y_binary_train, y_surv_train = process_dataframe(df_train)
    X_val_df, y_binary_val, y_surv_val = process_dataframe(df_val)
    X_test_df, y_binary_test, y_surv_test = process_dataframe(df_test)

    # 在训练集上拟合Scaler，并应用于所有数据集
    scaler = StandardScaler()
    X_train_df_filled = X_train_df.fillna(X_train_df.median())
    X_train = scaler.fit_transform(X_train_df_filled)
    
    X_val_df_filled = X_val_df.fillna(X_train_df.median()) # 使用训练集的中位数填充
    X_val = scaler.transform(X_val_df_filled)

    X_test_df_filled = X_test_df.fillna(X_train_df.median()) # 使用训练集的中位数填充
    X_test = scaler.transform(X_test_df_filled)

    print("从已划分的文件加载并预处理数据完成。")
    print(f"训练集: {len(X_train)}例, 验证集: {len(X_val)}例, 测试集: {len(X_test)}例")

    return (X_train, y_binary_train, y_surv_train, X_train_df,
            X_val, y_binary_val, y_surv_val, X_val_df,
            X_test, y_binary_test, y_surv_test, X_test_df,
            numeric_features, scaler)


# 训练各种模型
def train_models(X_train, y_binary_train, y_surv_train, X_train_df,
                 mlp_model_path=None, Transformer_MLP_model_path=None, transformer_model_path=None,
                 rnn_model_path=None, lstm_model_path=None, gru_model_path=None,
                 X_val=None, y_binary_val=None, y_surv_val=None,
                 tune_rf=True):
    """
    Train models with optional simple hyperparameter tuning.

    Parameters added:
    - X_val, y_binary_val, y_surv_val: optional validation sets used for hyperparam tuning
    - tune_rf: whether to run RandomizedSearchCV for RandomForest
    """
    models = {}

    # 1. 逻辑回归
    print("训练逻辑回归模型...")
    lr = LogisticRegression(random_state=43, max_iter=1000)
    lr.fit(X_train, y_binary_train)
    models['Logistic Regression'] = lr

    # 2. XGBoost
    print("训练XGBoost模型...")
    xgb_model = xgb.XGBClassifier(random_state=43, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_binary_train)
    models['XGBoost'] = xgb_model

    # 3. 随机森林
    print("训练随机森林模型...")
    rf = RandomForestClassifier(random_state=43)
    if tune_rf:
        # 简单随机搜索：n_estimators, max_depth, max_features, min_samples_split
        from sklearn.model_selection import RandomizedSearchCV
        param_dist = {
            'n_estimators': [50, 100, 200, 400],
            'max_depth': [None, 5, 10, 20],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [2, 5, 10]
        }
        rs = RandomizedSearchCV(rf, param_dist, n_iter=10, scoring='roc_auc', cv=3, random_state=43, n_jobs=-1)
        try:
            rs.fit(X_train, y_binary_train)
            best_rf = rs.best_estimator_
            print(f"RandomForest best params: {rs.best_params_}, best CV AUC: {rs.best_score_:.4f}")
            models['Random Forest'] = best_rf
        except Exception as e:
            print(f"RandomizedSearchCV failed, falling back to default RF: {e}")
            rf.set_params(n_estimators=100)
            rf.fit(X_train, y_binary_train)
            models['Random Forest'] = rf
    else:
        rf.set_params(n_estimators=100)
        rf.fit(X_train, y_binary_train)
        models['Random Forest'] = rf

    # 4. Cox比例风险模型
    print("训练CoxPH模型...")
    # 为 CoxPH 对原始特征做标准化（以提高数值稳定性）
    Cox_train_df = X_train_df.copy()
    Cox_train_df = Cox_train_df.fillna(Cox_train_df.median())
    scaler_cox = StandardScaler()
    Cox_train_df.loc[:, :] = scaler_cox.fit_transform(Cox_train_df)
    # 使用死亡事件(2年内死亡)而不是原来的存活标签
    Cox_train_df['event'] = y_surv_train['event'].astype(int)
    Cox_train_df['time'] = y_surv_train['time']

    # SVM
    print("训练SVM模型...")
    svm = SVC(probability=True, random_state=43)
    svm.fit(X_train, y_binary_train)
    models['SVM'] = svm

    # LightGBM
    print("训练LightGBM模型...")
    lgbm = lgb.LGBMClassifier(random_state=43)
    lgbm.fit(X_train, y_binary_train)
    models['LightGBM'] = lgbm

    # 如果提供了验证集，则基于验证集选取 penalizer（简单网格搜索）
    best_cph = None
    if X_val is not None and y_surv_val is not None:
        print("在验证集上为 CoxPH 选择 penalizer（网格）...")
        X_val_df_local = X_val.copy().fillna(X_val.median())
        X_val_df_local.loc[:, :] = scaler_cox.transform(X_val_df_local)
        best_ci = -np.inf
        best_pen = 0.0
        for pen in [0.0, 0.001, 0.01, 0.1, 1.0]:
            try:
                cph_tmp = CoxPHFitter(penalizer=pen)
                tmp_df = Cox_train_df.copy()
                cph_tmp.fit(tmp_df, duration_col='time', event_col='event')
                # 预测验证集风险并评估一致性指数
                pred_risk = cph_tmp.predict_partial_hazard(X_val_df_local)
                ci = concordance_index(y_surv_val['time'], -pred_risk.values.flatten(), y_surv_val['event'])
                print(f"penalizer={pen}, val concordance_index={ci:.4f}")
                if ci > best_ci:
                    best_ci = ci
                    best_pen = pen
                    best_cph = cph_tmp
            except Exception as e:
                print(f"CoxPH fit failed for penalizer={pen}: {e}")
        if best_cph is None:
            # 退回到无惩罚的模型
            best_cph = CoxPHFitter()
            best_cph.fit(Cox_train_df, duration_col='time', event_col='event')
            print("CoxPH: validation-based tuning failed, fitted default model.")
        else:
            print(f"CoxPH: selected penalizer={best_pen} with val CI={best_ci:.4f}")
            models['CoxPH'] = best_cph
        # 保存用于预测时的 scaler
        models['CoxPH_scaler'] = scaler_cox
    else:
        # 无验证集时使用默认拟合
        cph = CoxPHFitter()
        cph.fit(Cox_train_df, duration_col='time', event_col='event')
        models['CoxPH'] = cph
        models['CoxPH_scaler'] = scaler_cox

    # 5. 随机生存森林
    print("训练随机生存森林模型...")
    rsf = RandomSurvivalForest(random_state=43)
    if tune_rf: # Borrowing the tune_rf flag for RSF as well
        from sklearn.model_selection import RandomizedSearchCV
        from sksurv.metrics import concordance_index_censored

        rsf_params = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'min_samples_leaf': [1, 5, 10, 20]
        }
        # sksurv does not have a ready-made scorer for RandomizedSearchCV, so we define one.
        def sc_concordance_index(estimator, X, y):
            y_pred = estimator.predict(X)
            result = concordance_index_censored(y['event'], y['time'], y_pred)
            return result[0]

        rsf_search = RandomizedSearchCV(rsf, rsf_params, n_iter=10, scoring=sc_concordance_index, cv=3, random_state=43, n_jobs=-1)
        try:
            rsf_search.fit(X_train, y_surv_train)
            best_rsf = rsf_search.best_estimator_
            print(f"RandomSurvivalForest best params: {rsf_search.best_params_}, best CV C-index: {rsf_search.best_score_:.4f}")
            models['Random Survival Forest'] = best_rsf
        except Exception as e:
            print(f"RSF RandomizedSearchCV failed, falling back to default RSF: {e}")
            rsf.set_params(n_estimators=100)
            rsf.fit(X_train, y_surv_train)
            models['Random Survival Forest'] = rsf
    else:
        rsf.set_params(n_estimators=100)
        rsf.fit(X_train, y_surv_train)
        models['Random Survival Forest'] = rsf

    # 6. MLP模型
    if mlp_model_path:
        print(f"加载MLP模型: {mlp_model_path}")
        mlp = load_model(mlp_model_path, compile=False)
        models['MLP'] = mlp

    # 7. Transformer_MLP模型
    if Transformer_MLP_model_path:
        print(f"加载Transformer_MLP模型: {Transformer_MLP_model_path}")
        Transformer_MLP = load_model(Transformer_MLP_model_path, custom_objects={'TransformerEncoder': TransformerEncoder}, compile=False)
        models['Transformer_MLP'] = Transformer_MLP

    # 8. Transformer模型
    if transformer_model_path:
        print(f"加载Transformer模型: {transformer_model_path}")
        transformer = load_model(transformer_model_path, custom_objects={'TransformerEncoder': TransformerEncoder}, compile=False)
        models['Transformer'] = transformer

    # 9. LSTM 模型
    if lstm_model_path:
        print(f"加载LSTM模型: {lstm_model_path}")
        lstm = load_model(lstm_model_path, compile=False)
        models['LSTM'] = lstm

    # 10. GRU 模型
    if gru_model_path:
        print(f"加载GRU模型: {gru_model_path}")
        gru = load_model(gru_model_path, compile=False)
        models['GRU'] = gru

    # 11. RNN 模型
    if rnn_model_path:
        print(f"加载RNN模型: {rnn_model_path}")
        rnn = load_model(rnn_model_path, compile=False)
        models['RNN'] = rnn

    return models


# 获取所有模型的预测概率
def get_predictions(models, X_data, X_data_df, is_mlp=False, y_surv=None):
    predictions = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            pred_proba = model.predict_proba(X_data)[:, 1]
            predictions[name] = pred_proba

        elif name == 'XGBoost':
            pred_proba = model.predict_proba(X_data)[:, 1]
            predictions[name] = pred_proba

        elif name == 'Random Forest':
            pred_proba = model.predict_proba(X_data)[:, 1]
            predictions[name] = pred_proba

        elif name == 'SVM':
            pred_proba = model.predict_proba(X_data)[:, 1]
            predictions[name] = pred_proba

        elif name == 'LightGBM':
            pred_proba = model.predict_proba(X_data)[:, 1]
            predictions[name] = pred_proba

        elif name in ['MLP', 'Transformer_MLP', 'Transformer', 'RNN', 'LSTM', 'GRU'] and is_mlp:
            # 深度学习模型：根据模型输入形状自适应二维或三维输入，避免形状不匹配
            X_in = X_data
            try:
                in_shape = getattr(model, 'input_shape', None)
                if isinstance(in_shape, list):
                    in_shape = in_shape[0]
                if in_shape is not None and len(in_shape) == 3:
                    # 期望 (None, timesteps, features)
                    timesteps = in_shape[1] if in_shape[1] is not None else 1
                    # 常见情况 timesteps=1；若>1，优先使用1步填充
                    X_in = np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))
                else:
                    # 期望 (None, features)
                    X_in = X_data
            except Exception:
                X_in = X_data

            # 预测，必要时尝试回退到另一种形状
            try:
                pred = model.predict(X_in, verbose=0)
            except ValueError:
                # 如果失败且当前是2D，则尝试3D；反之亦然
                if X_in.ndim == 2:
                    X_try = np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))
                else:
                    X_try = X_data
                pred = model.predict(X_try, verbose=0)

            pred_vec = pred[0].flatten() if isinstance(pred, list) else pred.flatten()

            # 对 Transformer_MLP ，将风险分数转换为用于ROC/DCA的伪概率
            if name == 'Transformer_MLP':
                risk = pred_vec
                rmin, rmax = np.min(risk), np.max(risk)
                if rmax > rmin:
                    proba = 1.0 - (risk - rmin) / (rmax - rmin)
                else:
                    proba = np.full_like(risk, 0.5, dtype=float)
                predictions[name] = proba
            else:
                predictions[name] = pred_vec

    return predictions


# 绘制所有模型的ROC曲线
def plot_roc_curves(predictions, y_true, dataset_name, save_name="roc_comparison"):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 8))

    for name, pred_proba in predictions.items():
        if y_true is None or len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, pred_proba)
        auc = roc_auc_score(y_true, pred_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.4f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves Comparison on {dataset_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    # 保存和显示
    plt.tight_layout()
    save_path = f'{save_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC对比图像已保存为: {save_path}")
    plt.show()


# --- 新增DCA曲线绘制功能 ---
def calculate_dca(y_true, y_pred_proba, thresholds=np.arange(0, 1.01, 0.01)):
    """手动计算DCA的净收益(Net Benefit)"""
    n = len(y_true)
    events = np.sum(y_true)
    net_benefit = []

    for p in thresholds:
        y_pred = (y_pred_proba >= p).astype(int)
        # 确保混淆矩阵是2x2
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # 避免除以零
        if (1 - p) == 0:
            nb = (tp / n) - (fp / n) * (p / (1 - p + 1e-10))
        else:
            nb = (tp / n) - (fp / n) * (p / (1 - p))
        net_benefit.append(nb)

    all_treat = [(events / n) - ((n - events) / n) * (p / (1 - p + 1e-10)) for p in thresholds]
    no_treat = [0.0 for _ in thresholds]

    return {
        'thresholds': thresholds,
        'net_benefit': np.array(net_benefit),
        'all_treat': np.array(all_treat),
        'no_treat': np.array(no_treat)
    }

def plot_dca_curves(predictions, y_true, save_name="dca_comparison"):
    """绘制DCA曲线"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 8))

    dca_results = {}
    for name, pred_proba in predictions.items():
        if len(np.unique(y_true)) < 2:
            continue
        dca_results[name] = calculate_dca(y_true, pred_proba)

    for name, result in dca_results.items():
        plt.plot(result['thresholds'], result['net_benefit'], lw=2, label=name)

    if dca_results:
        first_result = next(iter(dca_results.values()))
        plt.plot(first_result['thresholds'], first_result['all_treat'], 'k--', lw=2, label='Treat All')
        plt.plot(first_result['thresholds'], first_result['no_treat'], 'k:', lw=2, label='Treat None')

    plt.xlim([0, 1])
    plt.ylim([-0.2, 0.6])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis (DCA)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = f'{save_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"DCA图像已保存为: {save_path}")
    plt.show()


# --- 新增KM曲线绘制功能 ---
def plot_km_curves(predictions, y_surv, save_name_prefix="km_curve"):
    """
    为每个模型绘制KM生存曲线，包含HR, 95% CI, p-value, Median OS，以及风险表(Number at risk)。
    predictions: 字典 {model_name: pred_proba}
    y_surv: 结构化数组，包含 'time' 和 'event'
    """
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    from lifelines.plotting import add_at_risk_counts
    
    plt.rcParams["font.family"] = "Times New Roman"
    
    for name, pred_proba in predictions.items():
        # 使用 subplots 创建画布，预留空间给风险表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 跳过无效预测
        if pred_proba is None or np.isnan(pred_proba).any():
            print(f"跳过 {name} 的 KM 曲线：预测包含 NaN。")
            plt.close(fig)
            continue

        # 根据预测概率中位数将患者分为高风险和低风险组
        # 注意：pred_proba 是生存概率（值越大越好），所以 < median 是高风险
        try:
            median_risk = np.median(pred_proba)
        except Exception as e:
            print(f"无法计算 {name} 的中位数风险: {e}")
            plt.close(fig)
            continue
            
        high_risk_mask = pred_proba < median_risk
        low_risk_mask = pred_proba >= median_risk
        
        T = y_surv['time']
        E = y_surv['event'].astype(int)

        # 确保每组至少有一个样本
        if sum(high_risk_mask) == 0 or sum(low_risk_mask) == 0:
            print(f"跳过 {name} 的 KM 曲线：某一组样本为空（high={sum(high_risk_mask)}, low={sum(low_risk_mask)})。")
            plt.close(fig)
            continue

        # 计算 HR (High vs Low)
        # group=1 (High Risk), group=0 (Low Risk)
        groups = high_risk_mask.astype(int)
        df_cox = pd.DataFrame({'time': T, 'event': E, 'group': groups})
        
        hr_val = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        
        try:
            cph = CoxPHFitter()
            cph.fit(df_cox, duration_col='time', event_col='event')
            hr_val = cph.hazard_ratios_['group']
            # lifelines 的 confidence_intervals_ 返回的是 log(HR) 的 CI，需要取指数转换为 HR 的 CI
            ci = np.exp(cph.confidence_intervals_.loc['group'])
            ci_lower = ci[0]
            ci_upper = ci[1]
        except Exception as e:
            print(f"CoxPH fit failed: {e}")

        # Log-rank检验
        try:
            results = logrank_test(T[high_risk_mask], T[low_risk_mask], E[high_risk_mask], E[low_risk_mask])
            p_value = results.p_value
        except:
            p_value = np.nan

        # 绘图
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(T[high_risk_mask], event_observed=E[high_risk_mask], label='High Risk')
        kmf_high.plot_survival_function(ax=ax, ci_show=True)
        median_os_high = kmf_high.median_survival_time_

        kmf_low = KaplanMeierFitter()
        kmf_low.fit(T[low_risk_mask], event_observed=E[low_risk_mask], label='Low Risk')
        kmf_low.plot_survival_function(ax=ax, ci_show=True)
        median_os_low = kmf_low.median_survival_time_
        
        # 添加风险表 (Number at risk table)
        add_at_risk_counts(kmf_high, kmf_low, ax=ax)
        
        plt.title(f'{name}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        
        # 格式化中位生存时间，处理无穷大情况
        def format_median(val):
            if np.isinf(val): return "Not Reached"
            return f"{val:.1f}"

        # 在图中显示详细统计信息
        text_str = (
            f"Median OS(months):\n"
            f"-High Risk: {format_median(median_os_high)}\n"
            f"-Low Risk: {format_median(median_os_low)}\n"
            f"Log-rank P={p_value:.4f}\n"
            f"HR (High vs Low): {hr_val:.2f}\n"
            f"95% CI: {ci_lower:.2f}-{ci_upper:.2f}"
        )
        
        # 调整文本位置，避免被风险表遮挡
        plt.text(0.05, 0.25, text_str, transform=ax.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        save_path = f'{save_name_prefix}_{name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KM曲线已保存为: {save_path}")
        plt.show()


# --- 新增校准曲线绘制功能 ---
def plot_calibration_curves(predictions, y_true, save_name_prefix="calibration_curve"):
    """
    绘制校准曲线 (Calibration Curve) 并计算 Brier Score。
    predictions: 字典 {model_name: pred_proba}
    y_true: 真实的二分类标签 (0/1)
    """
    from sklearn.calibration import calibration_curve
    
    plt.rcParams["font.family"] = "Times New Roman"
    
    for name, prob in predictions.items():
        if prob is None or np.isnan(prob).any():
            print(f"跳过 {name} 的校准曲线：预测包含 NaN。")
            continue
            
        plt.figure(figsize=(8, 8))
        
        # 计算校准曲线数据点
        # n_bins=10 表示将预测概率分成10个区间
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob, n_bins=10)
        
        # 计算 Brier Score
        try:
            brier = brier_score_loss(y_true, prob)
            brier_str = f"Brier Score: {brier:.4f}"
        except:
            brier_str = "Brier Score: N/A"

        # 绘制曲线
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name}", color='blue', linewidth=2)
        
        # 绘制完美校准的对角线
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)
        
        plt.ylabel("Fraction of positives (Observed)", fontsize=12)
        plt.xlabel("Mean predicted value (Predicted)", fontsize=12)
        plt.title(f"Calibration Curve - {name}\n{brier_str}", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        save_path = f'{save_name_prefix}_{name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"校准曲线已保存为: {save_path}")
        plt.show()


# --- 新增：校准优化功能 (Platt Scaling & Isotonic Regression) ---
def optimize_calibration(val_probs, val_y, test_probs, test_y, model_name):
    """
    使用 Platt Scaling 和 Isotonic Regression 对模型概率进行校准优化。
    注意：Brier Score 越低越好。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    
    print(f"\n=== 正在对 {model_name} 进行校准优化 ===")
    print("提示：Brier Score 越低代表校准越好 (0=完美, 1=最差)")
    
    # 准备数据 (LogisticRegression 需要 2D 输入)
    X_fit = val_probs.reshape(-1, 1)
    y_fit = val_y
    X_test = test_probs.reshape(-1, 1)
    
    # 1. Platt Scaling (基于 Logistic Regression)
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(X_fit, y_fit)
    platt_probs = lr.predict_proba(X_test)[:, 1]
    
    # 2. Isotonic Regression (保序回归)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_probs, y_fit)
    iso_probs = iso.predict(test_probs)
    
    # 计算 Brier Scores
    brier_orig = brier_score_loss(test_y, test_probs)
    brier_platt = brier_score_loss(test_y, platt_probs)
    brier_iso = brier_score_loss(test_y, iso_probs)
    
    print(f"Original Brier Score:       {brier_orig:.4f}")
    print(f"Platt Scaling Brier Score:  {brier_platt:.4f} ({(brier_platt-brier_orig)/brier_orig*100:.1f}%)")
    print(f"Isotonic Regression Brier:  {brier_iso:.4f} ({(brier_iso-brier_orig)/brier_orig*100:.1f}%)")
    
    # 绘图对比
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    
    # Original
    frac_orig, mean_orig = calibration_curve(test_y, test_probs, n_bins=10)
    plt.plot(mean_orig, frac_orig, "s-", label=f"Original (Brier={brier_orig:.4f})")
    
    # Platt
    frac_platt, mean_platt = calibration_curve(test_y, platt_probs, n_bins=10)
    plt.plot(mean_platt, frac_platt, "o-", label=f"Platt Scaling (Brier={brier_platt:.4f})")
    
    # Isotonic
    frac_iso, mean_iso = calibration_curve(test_y, iso_probs, n_bins=10)
    plt.plot(mean_iso, frac_iso, "^-", label=f"Isotonic (Brier={brier_iso:.4f})")
    
    plt.ylabel("Fraction of positives (Observed)")
    plt.xlabel("Mean predicted value (Predicted)")
    plt.title(f"Calibration Optimization - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = f"calibration_optimization_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"校准优化对比图已保存为: {save_path}")
    plt.show()


# 主程序
if __name__ == "__main__":
    # 配置参数
    train_path = 'train.xlsx'
    val_path = 'val.xlsx'
    test_path = 'test.xlsx'
    mlp_model_path = 'saved_models\\best_MLP_final6605.h5'
    Transformer_MLP_model_path = 'best_transformer_mlp_survival_final.h5'
    transformer_model_path = 'saved_models\\best_Transformer_final7034.h5'
    rnn_model_path = 'saved_models\\best_RNN_final6598.h5'
    lstm_model_path = 'saved_models\\best_LSTM_final6842.h5'
    gru_model_path = 'saved_models\\best_GRU_final6762.h5'

    # 加载已经划分好的数据
    print("=== 直接加载已划分的数据集 ===")
    try:
        df_train = pd.read_excel(train_path)
        df_val = pd.read_excel(val_path)
        df_test = pd.read_excel(test_path)
        print(f"成功加载: {train_path}, {val_path}, {test_path}")
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。请确保 train.xlsx, val.xlsx, 和 test.xlsx 文件在当前目录下。")
        exit()

    # 预处理已划分的数据
    (X_train, y_binary_train, y_surv_train, X_train_df,
     X_val, y_binary_val, y_surv_val, X_val_df,
     X_test, y_binary_test, y_surv_test, X_test_df,
     numeric_features, scaler) = preprocess_split_data(df_train, df_val, df_test)

    # 训练所有模型
    print("\n=== 训练所有模型 ===")
    models = train_models(X_train, y_binary_train, y_surv_train, X_train_df,
                          mlp_model_path, Transformer_MLP_model_path, transformer_model_path,
                          rnn_model_path, lstm_model_path, gru_model_path,
                          X_val=X_val_df, y_binary_val=y_binary_val, y_surv_val=y_surv_val,
                          tune_rf=True)

    # 获取所有模型的预测结果
    print("\n=== 生成预测结果 ===")
    train_predictions = get_predictions(models, X_train, X_train_df, is_mlp=True, y_surv=y_surv_train)
    val_predictions = get_predictions(models, X_val, X_val_df, is_mlp=True, y_surv=y_surv_val)
    test_predictions = get_predictions(models, X_test, X_test_df, is_mlp=True, y_surv=y_surv_test)

    # --- 打印评估指标 (训练/验证/测试)。对测试集启用 C-index 的95% CI计算（bootstrap） ---
    evaluate_and_print_metrics(train_predictions, y_binary_train, "训练集", y_surv=y_surv_train, compute_ci=False)
    evaluate_and_print_metrics(val_predictions, y_binary_val, "验证集", y_surv=y_surv_val, compute_ci=False)
    # 测试集上同时计算 C-index 与 95% CI（可能较慢，可通过 n_boot 参数调整）
    evaluate_and_print_metrics(test_predictions, y_binary_test, "测试集", y_surv=y_surv_test, compute_ci=True, n_boot=1000)

    # --- 分组绘制ROC曲线 ---
    print("\n=== 绘制ROC曲线对比 ===")

    # 定义两组模型
    group1_models = ['Transformer', 'Transformer_MLP', 'MLP', 'RNN', 'LSTM', 'GRU']
    group2_models = ['XGBoost', 'Random Forest', 'LightGBM', 'Logistic Regression', 'SVM']

    # 筛选模型的预测结果
    train_predictions_group1 = {name: pred for name, pred in train_predictions.items() if name in group1_models}
    test_predictions_group1 = {name: pred for name, pred in test_predictions.items() if name in group1_models}
    train_predictions_group2 = {name: pred for name, pred in train_predictions.items() if name in group2_models}
    test_predictions_group2 = {name: pred for name, pred in test_predictions.items() if name in group2_models}

    # 绘制第一组ROC曲线 - 测试集
    print("绘制第一组模型 (DL & SVM) 的测试集ROC曲线...")
    plot_roc_curves(test_predictions_group1, y_binary_test, "Test Set", save_name="roc_comparison_group1_test")

    # 绘制第一组ROC曲线 - 训练集
    print("绘制第一组模型 (DL & SVM) 的训练集ROC曲线...")
    plot_roc_curves(train_predictions_group1, y_binary_train, "Train Set", save_name="roc_comparison_group1_train")

    # 绘制第二组ROC曲线 - 测试集
    print("绘制第二组模型 (Traditional & Tree-based) 的测试集ROC曲线...")
    plot_roc_curves(test_predictions_group2, y_binary_test, "Test Set", save_name="roc_comparison_group2_test")

    # 绘制第二组ROC曲线 - 训练集
    print("绘制第二组模型 (Traditional & Tree-based) 的训练集ROC曲线...")
    plot_roc_curves(train_predictions_group2, y_binary_train, "Train Set", save_name="roc_comparison_group2_train")

    # --- 新增：分组绘制DCA曲线 ---
    print("\n=== 绘制DCA曲线对比 ===")
    
    # 绘制第一组DCA曲线
    print("绘制第一组模型 (DL & SVM) 的DCA曲线...")
    plot_dca_curves(test_predictions_group1, y_binary_test, save_name="dca_comparison_group1")

    # 绘制第二组DCA曲线
    print("绘制第二组模型 (Traditional & Tree-based) 的DCA曲线...")
    plot_dca_curves(test_predictions_group2, y_binary_test, save_name="dca_comparison_group2")

    # 评估并打印各模型在训练集和测试集上的性能指标（重复一次作为总结）
    print("\n=== 评估模型性能指标 ===")
    evaluate_and_print_metrics(train_predictions, y_binary_train, "训练集", y_surv=y_surv_train, compute_ci=False)
    evaluate_and_print_metrics(test_predictions, y_binary_test, "测试集", y_surv=y_surv_test, compute_ci=True, n_boot=1000)

    # --- 新增：绘制 Transformer_MLP 的 KM 曲线 ---
    print("\n=== 绘制 Transformer_MLP KM 曲线 ===")
    target_models = ['Transformer_MLP']
    
    # 训练集
    train_subset = {k: v for k, v in train_predictions.items() if k in target_models}
    if train_subset:
        plot_km_curves(train_subset, y_surv_train, save_name_prefix="km_train")
    
    # 测试集
    test_subset = {k: v for k, v in test_predictions.items() if k in target_models}
    if test_subset:
        plot_km_curves(test_subset, y_surv_test, save_name_prefix="km_test")

    # --- 新增：绘制 Transformer_MLP 的校准曲线 ---
    print("\n=== 绘制 Transformer_MLP 校准曲线 ===")
    # 训练集
    if 'Transformer_MLP' in train_predictions:
        plot_calibration_curves({'Transformer_MLP': train_predictions['Transformer_MLP']}, 
                                y_binary_train, save_name_prefix="calib_train")
    
    # 测试集
    if 'Transformer_MLP' in test_predictions:
        plot_calibration_curves({'Transformer_MLP': test_predictions['Transformer_MLP']}, 
                                y_binary_test, save_name_prefix="calib_test")

    # --- 新增：尝试校准优化 ---
    if 'Transformer_MLP' in val_predictions and 'Transformer_MLP' in test_predictions:
        optimize_calibration(val_predictions['Transformer_MLP'], y_binary_val,
                             test_predictions['Transformer_MLP'], y_binary_test,
                             "Transformer_MLP")