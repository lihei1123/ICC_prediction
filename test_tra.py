import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sksurv.util import Surv
import warnings
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
import os
import joblib

# 数据加载和预处理函数
def preprocess_split_data(df_train, df_val, df_test, features_to_use=None):
    """
    对已经划分好的训练、验证、测试集进行预处理。
    Scaler在训练集上拟合，并应用于所有数据集。
    """
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    if features_to_use is not None:
        selected_features = features_to_use
    else:
        selected_features = ['AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
                             'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
                             'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
                             'ALB', 'Nerve invasion']
    numeric_features = [f for f in selected_features if f in df_combined.columns]

    def process_dataframe(df):
        df['OS_months'] = pd.to_numeric(df['OS'], errors='coerce').clip(0, 180)
        # 二分类：是否存活超过2年（1=存活>24个月，0=≤24个月）
        df['survive_2years'] = (df['OS_months'] > 24).astype(int)
        y_binary = df['survive_2years'].values
        # 生存分析事件：是否在2年内死亡（1=≤24个月发生事件，0=截尾/未在2年内死亡）
        death_within_2y = (df['OS_months'] <= 24).astype(int)
        y_surv = Surv.from_arrays(event=death_within_2y.values, time=df['OS_months'].values)
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

    print("从已划分的文件加载并预处理数据完成。")
    print(f"训练集: {len(X_train)}例, 验证集: {len(X_val)}例, 测试集: {len(X_test)}例")

    return (X_train, y_binary_train, y_surv_train, X_train_df,
            X_val, y_binary_val, y_surv_val, X_val_df,
            X_test, y_binary_test, y_surv_test, X_test_df,
            numeric_features, scaler)


# 训练传统模型
def train_models(X_train, y_binary_train,
                 tune_rf=True):
    """
    Train traditional ML models.
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

    # 4. SVM
    print("训练SVM模型...")
    svm = SVC(probability=True, random_state=43)
    svm.fit(X_train, y_binary_train)
    models['SVM'] = svm

    # 5. LightGBM
    print("训练LightGBM模型...")
    lgbm = lgb.LGBMClassifier(random_state=43)
    lgbm.fit(X_train, y_binary_train)
    models['LightGBM'] = lgbm

    return models


def make_base_estimators(random_state=43):
    """Return a dict of unfitted sklearn-compatible estimators matching train_models defaults."""
    estimators = {}
    estimators['Logistic Regression'] = LogisticRegression(random_state=random_state, max_iter=1000)
    estimators['XGBoost'] = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    estimators['Random Forest'] = RandomForestClassifier(random_state=random_state, n_estimators=100)
    estimators['SVM'] = SVC(probability=True, random_state=random_state)
    estimators['LightGBM'] = lgb.LGBMClassifier(random_state=random_state)
    return estimators


# 保存/加载传统模型的路径配置与工具函数
traditional_paths = {
    'SVM': 'saved_models/svm_model.joblib',  # 根目录下示例
    'Logistic Regression': 'saved_models/logistic_regression.joblib',
    'XGBoost': 'saved_models/xgboost.joblib',
    'Random Forest': 'saved_models/random_forest.joblib',
    'LightGBM': 'saved_models/lightgbm.joblib'
}


def load_traditional_models(paths):
    """尝试从磁盘加载所有路径中列出的模型；如果全部存在并成功加载则返回 dict，否则返回 None。"""
    loaded = {}
    for name, p in paths.items():
        try:
            if os.path.exists(p):
                loaded[name] = joblib.load(p)
            else:
                # 如果任意一个模型文件不存在，则放弃全部加载
                return None
        except Exception as e:
            print(f"加载模型 {p} 失败: {e}")
            return None
    print("已从磁盘加载所有传统模型（跳过训练）。")
    return loaded


def save_traditional_models(models, paths):
    """将训练好的模型保存到指定路径（按 paths 字典）。"""
    for name, model in models.items():
        path = paths.get(name)
        if not path:
            continue
        # 确保目录存在
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        try:
            joblib.dump(model, path)
            print(f"Saved {name} -> {path}")
        except Exception as e:
            print(f"保存模型 {name} 到 {path} 失败: {e}")


# 获取所有模型的预测概率
def get_predictions(models, X_data):
    predictions = {}

    for name, model in models.items():
        # All models here use predict_proba
        pred_proba = model.predict_proba(X_data)[:, 1]
        predictions[name] = pred_proba

    return predictions

def _bootstrap_cindex(risk_scores, y_surv, n_boot=1000, random_state=43):
    """使用自助法(bootstrap)基于 sksurv 的 concordance_index_censored 估计C-index的95%置信区间。
    传入的是风险分数（数值越大代表风险越高）。返回 (c_index, ci_lower, ci_upper)
    """
    from sksurv.metrics import concordance_index_censored
    rng = np.random.RandomState(random_state)
    times = np.asarray(y_surv['time'])
    events = np.asarray(y_surv['event']).astype(bool)
    preds = np.asarray(risk_scores).flatten()

    # 原始 C-index
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
    out = {}
    if y_true is None or len(np.unique(y_true)) < 2:
        return None
    try:
        out['AUC'] = float(roc_auc_score(y_true, pred_proba))
    except Exception:
        out['AUC'] = np.nan
    y_pred = (np.asarray(pred_proba) >= 0.5).astype(int)
    out['Accuracy'] = float(accuracy_score(y_true, y_pred))
    out['Sensitivity'] = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out['Specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    out['PPV'] = float(precision_score(y_true, y_pred, zero_division=0))
    out['NPV'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    try:
        out['Brier'] = float(brier_score_loss(y_true, pred_proba))
    except Exception:
        out['Brier'] = np.nan
    return out


def evaluate_and_print_metrics(predictions, y_true, dataset_name, y_surv=None, compute_ci=False, n_boot=1000):
    print(f"\n--- {dataset_name} 评估结果 ---")
    headers = ['Model', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Brier']
    print(f"{headers[0]:<25} | {headers[1]:<6} | {headers[2]:<8} | {headers[3]:<10} | {headers[4]:<11} | {headers[5]:<6} | {headers[6]:<6} | {headers[7]:<7}")
    print("-" * 110)

    for name, pred_proba in predictions.items():
        metrics = compute_metrics_for_model(pred_proba, y_true)
        if metrics is None:
            print(f"{name:<25} | {'N/A (insufficient labels)':<85}")
            continue
        print(f"{name:<25} | {metrics['AUC']:<6.4f} | {metrics['Accuracy']:<8.4f} | {metrics['Sensitivity']:<10.4f} | {metrics['Specificity']:<11.4f} | {metrics['PPV']:<6.4f} | {metrics['NPV']:<6.4f} | {metrics['Brier']:<7.4f}")

    # C-index
    if y_surv is not None:
        print("\nC-index on this dataset:")
        print(f"{'Model':<25} | {'C-index':<8} | {'95% CI':<20}")
        print("-" * 60)
        for name, pred_proba in predictions.items():
            try:
                # 从 y_surv 获取时间与事件
                times = np.asarray(y_surv['time'])
                events = np.asarray(y_surv['event']).astype(bool)
                # 风险分数定义：
                # - CoxPH、RSF 等直接输出风险/危害度 -> 直接使用
                # - 其它二分类概率模型 -> 风险 = 1 - 概率（这里概率代表存活>2年）
                if name in ['CoxPH', 'Random Survival Forest']:
                    risk_scores = np.asarray(pred_proba).flatten()
                else:
                    risk_scores = 1 - np.asarray(pred_proba).flatten()
                from sksurv.metrics import concordance_index_censored
                c = concordance_index_censored(events, times, risk_scores)[0]

                if compute_ci:
                    try:
                        # _bootstrap_cindex 也需要传入风险分数
                        c_orig, lower, upper = _bootstrap_cindex(risk_scores, y_surv, n_boot=n_boot)
                        print(f"{name:<25} | {c_orig:<8.4f} | ({lower:.4f}, {upper:.4f})")
                    except Exception as e:
                        print(f"{name:<25} | {c:<8.4f} | CI计算失败: {e}")
                else:
                    print(f"{name:<25} | {c:<8.4f} | {'N/A':<20}")
            except Exception as e:
                print(f"{name:<25} | Error computing C-index: {e}")


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


# 绘制所有模型的ROC曲线
def plot_roc_curves(train_predictions, test_predictions, y_train, y_test, save_name="roc_comparison", plot_train=True):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 8))

    # Plotting test curves first to have them appear more prominently
    for name, pred_proba in test_predictions.items():
        if len(np.unique(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        auc = roc_auc_score(y_test, pred_proba)
        plt.plot(fpr, tpr, lw=2, linestyle='-', label=f'{name} (Test, AUC={auc:.4f})')

    if plot_train:
        for name, pred_proba in train_predictions.items():
            if y_train is None or len(np.unique(y_train)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_train, pred_proba)
            auc = roc_auc_score(y_train, pred_proba)
            plt.plot(fpr, tpr, lw=1, linestyle='--', label=f'{name} (Train, AUC={auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison (Traditional Models)', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = f'{save_name}3.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC对比图像已保存为: {save_path}")
    plt.show()


def run_feature_selection_experiment(df_train, df_val, df_test, all_features):
    """
    运行特征选择实验，评估不同数量的特征对模型性能的影响。
    """
    results = []
    
    # 遍历不同数量的特征
    for n_features in range(1, len(all_features) + 1):
        current_features = all_features[:n_features]
        print(f"\n=== 正在使用 {n_features} 个特征进行测试 ===")
        print(f"使用的特征: {current_features}")

        # 1. 数据预处理
        (X_train, y_binary_train, _, _,
         X_val, y_binary_val, _, _,
         X_test, y_binary_test, _, _,
         _, _) = preprocess_split_data(df_train.copy(), df_val.copy(), df_test.copy(), features_to_use=current_features)

        # 合并训练集和验证集
        X_train_full = np.concatenate((X_train, X_val), axis=0)
        y_binary_train_full = np.concatenate((y_binary_train, y_binary_val), axis=0)

        # 2. 对每个模型做 5 折交叉验证（在 train+val 上）并记录 cv mean/std，然后在 train+val 上训练最终模型并评估 test
        estimators = make_base_estimators()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result_row = {'n_features': n_features}
        for name, est in estimators.items():
            # 如果训练标签只有一个类别，则无法做 CV
            if len(np.unique(y_binary_train_full)) < 2:
                cv_mean = np.nan
                cv_std = np.nan
                print(f"跳过 {name} 的 CV（训练集只有单一标签）。")
            else:
                try:
                    scores = cross_val_score(est, X_train_full, y_binary_train_full, cv=skf, scoring='roc_auc', n_jobs=-1)
                    cv_mean = np.nanmean(scores)
                    cv_std = np.nanstd(scores)
                except Exception as e:
                    print(f"对模型 {name} 在 k={n_features} 做 CV 失败: {e}")
                    cv_mean = np.nan
                    cv_std = np.nan

            # 记录 CV 结果
            result_row[f"{name}_cv_mean"] = cv_mean
            result_row[f"{name}_cv_std"] = cv_std

            # 用 train+val 训练最终模型并评估 test
            try:
                est.fit(X_train_full, y_binary_train_full)
                pred_proba = None
                if len(np.unique(y_binary_test)) > 1:
                    try:
                        pred_proba = est.predict_proba(X_test)[:, 1]
                    except Exception:
                        # 有的模型（极少数）可能没有 predict_proba：退回到 decision_function
                        try:
                            pred_scores = est.decision_function(X_test)
                            pred_proba = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min())
                        except Exception:
                            pred_proba = None
                if pred_proba is not None:
                    test_auc = roc_auc_score(y_binary_test, pred_proba)
                else:
                    test_auc = np.nan
            except Exception as e:
                print(f"训练/评估模型 {name} 失败: {e}")
                test_auc = np.nan

            result_row[name] = test_auc
            print(f"模型: {name}, 特征数: {n_features}, CV mean AUC={cv_mean if not np.isnan(cv_mean) else 'N/A'} , test AUC={test_auc if not np.isnan(test_auc) else 'N/A'}")

        results.append(result_row)

    return pd.DataFrame(results)

def plot_feature_selection_results(results_df):
    """
    绘制特征数量与模型性能的关系图。
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(14, 8))
    
    for model_name in results_df.columns:
        # Skip CV summary columns
        if model_name == 'n_features' or model_name.endswith('_cv_mean') or model_name.endswith('_cv_std'):
            continue
        plt.plot(results_df['n_features'], results_df[model_name], marker='o', linestyle='-', label=model_name)

    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Test Set AUC', fontsize=12)
    plt.title('Model Performance vs. Number of Features', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(range(1, len(results_df) + 1))
    plt.tight_layout()
    
    save_path = 'feature_selection_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n特征选择性能图已保存为: {save_path}")
    plt.show()

# 主程序
if __name__ == "__main__":
    train_path = 'train.xlsx'
    val_path = 'val.xlsx'
    test_path = 'test.xlsx'

    print("=== 直接加载已划分的数据集 ===")
    try:
        df_train = pd.read_excel(train_path)
        df_val = pd.read_excel(val_path)
        df_test = pd.read_excel(test_path)
        print(f"成功加载: {train_path}, {val_path}, {test_path}")
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。请确保 train.xlsx, val.xlsx, 和 test.xlsx 文件在当前目录下。")
        exit()

    (X_train, y_binary_train, y_surv_train, X_train_df,
     X_val, y_binary_val, y_surv_val, X_val_df,
     X_test, y_binary_test, y_surv_test, X_test_df,
     numeric_features, scaler) = preprocess_split_data(df_train, df_val, df_test)

    # print("\n=== 开始特征选择实验 ===")
    # # 原始的特征列表，按重要性降序排列
    # all_features_ordered = ['AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
    #                          'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
    #                          'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
    #                          'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
    #                          'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
    #                          'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
    #                          'Cardiopulmonary disease', 'Surgical type', 'Hepatitis']
    
    # # 重新加载数据以避免在循环中出现问题
    # df_train_orig = pd.read_excel(train_path)
    # df_val_orig = pd.read_excel(val_path)
    # df_test_orig = pd.read_excel(test_path)

    # results_df = run_feature_selection_experiment(df_train_orig, df_val_orig, df_test_orig, all_features_ordered)
    
    # print("\n=== 特征选择实验结果 ===")
    # # 原始的宽表格（每行对应一个 n_features，列包含每个模型的 test AUC）
    # print(results_df)

        # print("\n--- 按 k 汇总的模型 Test AUC 表（每行 k，对应各模型） ---")
    # # 选取模型列（排除 CV 统计列）
    # model_cols = [c for c in results_df.columns if c != 'n_features' and not c.endswith('_cv_mean') and not c.endswith('_cv_std')]
    # # 构建按 n_features 排序的表格，并将 n_features 设为索引
    # try:
    #     table = results_df[["n_features"] + model_cols].sort_values('n_features').set_index('n_features')
    #     # 打印表头
    #     header = "k\t" + "\t".join(model_cols)
    #     print(header)
    #     # 打印每一行，保留 6 位小数
    #     for idx, row in table.iterrows():
    #         vals = [f"{row[c]:.6f}" if (pd.notna(row[c])) else "N/A" for c in model_cols]
    #         print(f"{int(idx):2d}\t" + "\t".join(vals))
    # except Exception as e:
    #     print('生成按 k 汇总表失败:', e)

    # # 绘制 mean_auc 与 max_auc 随 k 变化的图（与 shaixuan2.py 风格一致）
    # try:
    #     # 计算每个 k 的 mean 和 max（跨模型）
    #     summary_df = table.copy()
    #     summary_df['mean_auc'] = summary_df[model_cols].mean(axis=1, skipna=True)
    #     summary_df['max_auc'] = summary_df[model_cols].max(axis=1, skipna=True)

    #     plt.rcParams["font.family"] = "Times New Roman"
    #     plt.figure(figsize=(10, 8))
    #     plt.plot(summary_df.index, summary_df['mean_auc'], marker='o', lw=2, linestyle='-', label='Mean test AUC (across models)')
    #     plt.plot(summary_df.index, summary_df['max_auc'], marker='s', lw=2, linestyle='-', label='Best test AUC (across models)')
    #     plt.xlabel('Number of Features (k)', fontsize=12)
    #     plt.ylabel('AUC on test set', fontsize=12)
    #     plt.title('Feature selection — Traditional models (mean & max AUC)', fontsize=14)
    #     plt.grid(alpha=0.3)
    #     plt.legend(loc='lower right', fontsize=10)
    #     plt.tight_layout()
    #     png = 'feature_selection_traditional_mean_max.png'
    #     plt.savefig(png, dpi=300, bbox_inches='tight')
    #     print(f"Mean/Max AUC 曲线已保存为: {png}")
    #     plt.show()
    # except Exception as e:
    #     print('绘制 mean/max AUC 曲线失败:', e)

    # plot_feature_selection_results(results_df)

    # # 找到每个模型的最佳特征数（只看 test AUC 列，跳过 CV 列）
    # print("\n--- 各模型最佳表现 ---")
    # model_cols = [c for c in results_df.columns if c != 'n_features' and not c.endswith('_cv_mean') and not c.endswith('_cv_std')]
    # for model_name in model_cols:
    #     try:
    #         best_score = results_df[model_name].max()
    #         best_n_features = int(results_df.loc[results_df[model_name].idxmax(), 'n_features'])
    #         print(f"模型: {model_name}, 最佳特征数: {best_n_features}, 最高 AUC: {best_score:.4f}")
    #     except Exception as e:
    #         print(f"无法计算模型 {model_name} 的最佳表现: {e}")

    # 如果你还想运行一次使用所有特征的原始流程，可以取消下面的注释
    print("=== 训练传统模型（使用所有特征） ===")
    # 合并训练集和验证集用于最终模型训练
    X_train_full = np.concatenate((X_train, X_val), axis=0)
    y_binary_train_full = np.concatenate((y_binary_train, y_binary_val), axis=0)

    # 优先尝试从磁盘加载已保存的模型（如果全部存在则跳过训练）
    models = load_traditional_models(traditional_paths)
    if models is None:
        models = train_models(X_train_full, y_binary_train_full, tune_rf=True)
        # 训练完成后保存模型
        try:
            save_traditional_models(models, traditional_paths)
        except Exception as e:
            print(f"保存传统模型时发生错误: {e}")

    print("\n=== 生成预测结果 ===")
    train_predictions = get_predictions(models, X_train_full)
    val_predictions = get_predictions(models, X_val)
    test_predictions = get_predictions(models, X_test)

    # 合并 survival 结构用于在训练集上计算 C-index（如果需要）
    try:
        y_surv_train_full = np.concatenate((y_surv_train, y_surv_val))
    except Exception:
        y_surv_train_full = None

    # 打印评估指标
    evaluate_and_print_metrics(train_predictions, y_binary_train_full, '训练集', y_surv=y_surv_train_full, compute_ci=False)
    evaluate_and_print_metrics(val_predictions, y_binary_val, '验证集', y_surv=y_surv_val, compute_ci=False)
    evaluate_and_print_metrics(test_predictions, y_binary_test, '测试集', y_surv=y_surv_test, compute_ci=True, n_boot=1000)

    # 绘制 DCA 曲线
    print("\n=== 绘制DCA曲线（传统模型） ===")
    plot_dca_curves(test_predictions, y_binary_test, save_name="dca_comparison_traditional_models")

    print("\n=== 绘制ROC曲线对比 ===")
    plot_roc_curves(train_predictions, test_predictions, y_binary_train_full, y_binary_test, 
                    save_name="roc_comparison_traditional_models", plot_train=True)
