import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from sksurv.util import Surv
import warnings
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

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
                             'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
                             'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
                             'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
                             'Cardiopulmonary disease', 'Surgical type', 'Hepatitis']
    numeric_features = [f for f in selected_features if f in df_combined.columns]

    def process_dataframe(df):
        df['OS_months'] = pd.to_numeric(df['OS'], errors='coerce').clip(0, 180)
        df['survive_2years'] = (df['OS_months'] > 24).astype(int)
        y_binary = df['survive_2years'].values
        y_surv = Surv.from_arrays(event=df['survive_2years'].values, time=df['OS_months'].values)
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


def make_base_estimators():
    """Return a dict of unfitted sklearn-compatible estimators matching train_models defaults.
    注意：不在此处固定 random_state，改为在重复 CV 内部为每次重复设置，以保证模型间差异与重复间独立性。
    """
    estimators = {}
    estimators['Logistic Regression'] = LogisticRegression(max_iter=1000)
    estimators['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    estimators['Random Forest'] = RandomForestClassifier(n_estimators=100)
    estimators['SVM'] = SVC(probability=True)
    estimators['LightGBM'] = lgb.LGBMClassifier()
    return estimators


# 获取所有模型的预测概率
def get_predictions(models, X_data):
    predictions = {}

    for name, model in models.items():
        # All models here use predict_proba
        pred_proba = model.predict_proba(X_data)[:, 1]
        predictions[name] = pred_proba

    return predictions


# ----------------- 评估与 C-index bootstrap -----------------
def _bootstrap_cindex(pred_proba, y_surv, n_boot=1000, random_state=43):
    """使用 bootstrap 估计 C-index 的 95% CI，返回 (c_index, lower, upper)。"""
    from lifelines.utils import concordance_index as lifelines_concordance
    rng = np.random.RandomState(random_state)
    times = np.array(y_surv['time'])
    events = np.array(y_surv['event'])
    preds = np.asarray(pred_proba).flatten()

    # 原始 c-index
    try:
        c_orig = lifelines_concordance(times, preds, events)
    except Exception:
        from sksurv.metrics import concordance_index_censored
        c_orig = concordance_index_censored(events, times, preds)[0]

    n = len(preds)
    boot_scores = []
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            c = lifelines_concordance(times[idx], preds[idx], events[idx])
        except Exception:
            from sksurv.metrics import concordance_index_censored
            c = concordance_index_censored(events[idx], times[idx], preds[idx])[0]
        boot_scores.append(c)

    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    return c_orig, lower, upper


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
    """对每个模型输出 AUC/Accuracy/Sensitivity/Specificity/PPV/NPV/Brier，并可选对 C-index 做 bootstrap CI。"""
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

    # C-index（如果提供 survival 信息）
    if y_surv is not None:
        print("\nC-index on this dataset:")
        print(f"{'Model':<25} | {'C-index':<8} | {'95% CI':<20}")
        print("-" * 60)
        for name, pred_proba in predictions.items():
            try:
                from lifelines.utils import concordance_index as lifelines_concordance
                times = np.array(y_surv['time'])
                events = np.array(y_surv['event'])
                risk_scores = 1 - np.asarray(pred_proba).flatten()
                
                try:
                    c = lifelines_concordance(times, risk_scores, events)
                except Exception:
                    from sksurv.metrics import concordance_index_censored
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


def run_feature_selection_experiment(df_train, df_val, df_test, all_features, n_repeats=1):
    """
    运行特征选择实验，评估不同数量的特征对模型性能的影响。
    对于每个特征数 k，执行 n_repeats 次重复：每次重新做 5 折 StratifiedKFold CV，记录该次的验证 AUC 均值；
    最后对 n_repeats 次的 CV 均值再求总体均值 ± 标准差，用于选择最佳 k。
    返回的 DataFrame 中：
      {Model}_cv_mean  表示 n_repeats 次重复的 CV 均值的平均值
      {Model}_cv_repeats_std 表示 n_repeats 次重复的 CV 均值的标准差（反映重复间稳定性）
      {Model} 仍保留一次基于 train+val 训练后在 test 上的 AUC（仅参考，可忽略）
    """
    from sklearn.base import clone
    results = []

    for n_features in range(1, len(all_features) + 1):
        current_features = all_features[:n_features]
        print(f"\n=== 正在使用 {n_features} 个特征进行测试 ===")
        print(f"使用的特征: {current_features}")

        # 数据预处理
        (X_train, y_binary_train, _, X_train_df,
         X_val, y_binary_val, _, X_val_df,
         X_test, y_binary_test, _, X_test_df,
         numeric_features, _) = preprocess_split_data(df_train.copy(), df_val.copy(), df_test.copy(), features_to_use=current_features)

        # 合并训练+验证
        X_train_full = np.concatenate((X_train, X_val), axis=0)  # 仅用于保持与原有输出一致，不用于CV预处理
        y_binary_train_full = np.concatenate((y_binary_train, y_binary_val), axis=0)
        X_raw_full = pd.concat([X_train_df, X_val_df], axis=0, ignore_index=True)

        estimators = make_base_estimators()
        result_row = {'n_features': n_features}

        for name, base_est in estimators.items():
            if len(np.unique(y_binary_train_full)) < 2:
                cv_mean_across_repeats = np.nan
                cv_std_across_repeats = np.nan
                print(f"跳过 {name} 的 CV（训练集只有单一标签）。")
                test_auc = np.nan
            else:
                repeat_means = []
                for r in range(n_repeats):
                    random_state = 42 + r
                    np.random.seed(random_state)
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

                    est = clone(base_est)
                    try:
                        params = est.get_params()
                        if 'random_state' in params:
                            est.set_params(random_state=random_state)
                    except Exception:
                        pass
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('clf', est)
                    ])

                    try:
                        scores = cross_val_score(pipeline, X_raw_full, y_binary_train_full, cv=skf, scoring='roc_auc', n_jobs=-1)
                        repeat_means.append(np.nanmean(scores))
                    except Exception as e:
                        print(f"  重复 {r+1}/{n_repeats} - {name} CV 失败: {e}")
                        repeat_means.append(np.nan)
                cv_mean_across_repeats = np.nanmean(repeat_means)
                cv_std_across_repeats = np.nanstd(repeat_means)

                try:
                    final_est = clone(base_est)
                    try:
                        params = final_est.get_params()
                        if 'random_state' in params:
                            final_est.set_params(random_state=2025)
                    except Exception:
                        pass
                    final_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('clf', final_est)
                    ])
                    final_pipeline.fit(X_raw_full, y_binary_train_full)
                    if len(np.unique(y_binary_test)) > 1:
                        try:
                            pred_proba = final_pipeline.predict_proba(X_test_df)[:, 1]
                        except Exception:
                            pred_scores = final_pipeline.decision_function(X_test_df)
                            pred_proba = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min())
                        test_auc = roc_auc_score(y_binary_test, pred_proba)
                    else:
                        test_auc = np.nan
                except Exception as e:
                    print(f"  最终模型训练/测试评估失败: {name}, 错误: {e}")
                    test_auc = np.nan

                print(f"模型: {name}, 特征数: {n_features}, 5次重复CV均值={cv_mean_across_repeats if not np.isnan(cv_mean_across_repeats) else 'N/A'} ± {cv_std_across_repeats if not np.isnan(cv_std_across_repeats) else 'N/A'}, test AUC={test_auc if not np.isnan(test_auc) else 'N/A'}")

            result_row[f"{name}_cv_mean"] = cv_mean_across_repeats
            result_row[f"{name}_cv_repeats_std"] = cv_std_across_repeats
            result_row[name] = test_auc

        results.append(result_row)

    return pd.DataFrame(results)

def plot_feature_selection_results(results_df):
    """
    绘制特征数量与模型性能的关系图（使用 5 折 CV 的验证集 AUC 均值）。
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(14, 8))

    # 仅绘制 *_cv_mean 列（各模型在 train+val 上 5 折交叉验证得到的验证集 AUC 均值）
    cv_mean_cols = [c for c in results_df.columns if c.endswith('_cv_mean')]
    for col in cv_mean_cols:
        base_name = col.replace('_cv_mean', '')
        plt.plot(results_df['n_features'], results_df[col], marker='o', linestyle='-', label=base_name)

    plt.xlabel('Number of Features (k)', fontsize=12)
    plt.ylabel('Validation AUC (5-fold CV)', fontsize=12)
    plt.title('CV mean AUC vs. Number of Features (Traditional Models)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(range(1, len(results_df) + 1))
    plt.tight_layout()

    save_path = 'feature_selection_performance_cv.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n特征选择性能图（CV mean AUC）已保存为: {save_path}")
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

    print("\n=== 开始特征选择实验 ===")
    # 原始的特征列表，按重要性降序排列
    all_features_ordered = ['AC', 'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
                             'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
                             'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
                             'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
                             'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
                             'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
                             'Cardiopulmonary disease', 'Surgical type', 'Hepatitis']
    
    # 重新加载数据以避免在循环中出现问题
    df_train_orig = pd.read_excel(train_path)
    df_val_orig = pd.read_excel(val_path)
    df_test_orig = pd.read_excel(test_path)

    results_df = run_feature_selection_experiment(df_train_orig, df_val_orig, df_test_orig, all_features_ordered, n_repeats=5)
    
    print("\n=== 特征选择实验结果 ===")
    # 打印完整结果（包含 CV 均值与测试集 AUC），便于完整留痕
    print(results_df)

    # 打印一个简洁的表格：每一行对应 k（n_features），列为各模型的 验证集 AUC（5 折 CV 均值）
    print("\n--- 按 k 汇总的模型 验证集 AUC（5 折 CV 5次重复后均值）表 ---")
    cv_model_cols = [c for c in results_df.columns if c.endswith('_cv_mean')]
    # 构建按 n_features 排序的表格，并将 n_features 设为索引
    try:
        cv_table = results_df[["n_features"] + cv_model_cols].sort_values('n_features').set_index('n_features')
        # 生成用于显示的模型名称（去掉后缀）
        display_names = [c.replace('_cv_mean', '') for c in cv_model_cols]
        # 打印表头
        header = "k\t" + "\t".join(display_names)
        print(header)
        # 打印每一行，保留 6 位小数
        for idx, row in cv_table.iterrows():
            vals = [f"{row[c]:.6f}" if (pd.notna(row[c])) else "N/A" for c in cv_model_cols]
            print(f"{int(idx):2d}\t" + "\t".join(vals))
    except Exception as e:
        print('生成按 k 的 CV 均值汇总表失败:', e)

    # 绘制 mean_auc 与 max_auc 随 k 变化的图（基于 5 折 CV 的验证集 AUC 均值）
    try:
        # 使用 *_cv_mean 列构建用于汇总的表
        cv_model_cols = [c for c in results_df.columns if c.endswith('_cv_mean')]
        cv_table = results_df[["n_features"] + cv_model_cols].sort_values('n_features').set_index('n_features')

        # 计算每个 k 的 mean 和 max（跨模型）
        summary_df = cv_table.copy()
        summary_df['mean_auc'] = summary_df[cv_model_cols].mean(axis=1, skipna=True)
        summary_df['max_auc'] = summary_df[cv_model_cols].max(axis=1, skipna=True)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(10, 8))
        plt.plot(summary_df.index, summary_df['mean_auc'], marker='o', lw=2, linestyle='-', label='Mean CV AUC (across models)')
        plt.plot(summary_df.index, summary_df['max_auc'], marker='s', lw=2, linestyle='-', label='Best CV AUC (across models)')
        plt.xlabel('Number of Features (k)', fontsize=12)
        plt.ylabel('Validation AUC (5-fold CV)', fontsize=12)
        plt.title('Feature selection — Traditional models (CV mean & max AUC)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        png = 'feature_selection_traditional_mean_max_cv.png'
        plt.savefig(png, dpi=300, bbox_inches='tight')
        print(f"CV Mean/Max AUC 曲线已保存为: {png}")
        plt.show()
    except Exception as e:
        print('绘制 mean/max AUC 曲线失败:', e)

    plot_feature_selection_results(results_df)

    # 找到每个模型的最佳特征数（基于 5 次重复的 CV 均值）
    print("\n--- 各模型最佳表现（基于 5 次重复的 CV 均值） ---")
    cv_model_cols = [c for c in results_df.columns if c.endswith('_cv_mean')]
    for col in cv_model_cols:
        try:
            best_score = results_df[col].max()
            best_n_features = int(results_df.loc[results_df[col].idxmax(), 'n_features'])
            base_name = col.replace('_cv_mean', '')
            # 同时取该点的重复间标准差
            std_col = col.replace('_cv_mean', '_cv_repeats_std')
            std_val = results_df.loc[results_df[col].idxmax(), std_col] if std_col in results_df.columns else np.nan
            if not np.isnan(std_val):
                print(f"模型: {base_name}, 最佳特征数: {best_n_features}, CV均值: {best_score:.4f} ± {std_val:.4f}")
            else:
                print(f"模型: {base_name}, 最佳特征数: {best_n_features}, CV均值: {best_score:.4f}")
        except Exception as e:
            base_name = col.replace('_cv_mean', '')
            print(f"无法计算模型 {base_name} 的最佳表现: {e}")

    # 整体最佳 k（按跨模型 CV 均值的平均值最大）
    try:
        overall_mean_series = results_df[cv_model_cols].mean(axis=1, skipna=True)
        overall_best_idx = int(overall_mean_series.idxmax())
        overall_best_k = int(results_df.loc[overall_best_idx, 'n_features'])
        overall_best_mean = overall_mean_series.loc[overall_best_idx]
        print(f"\n整体最佳特征数 (按跨模型CV均值最大): k = {overall_best_k}, 平均CV AUC = {overall_best_mean:.4f}")
    except Exception as e:
        print(f"计算整体最佳特征数失败: {e}")

    # # 运行一次使用所有特征的原始流程，可以取消下面的注释
    # print("\n=== 训练传统模型（使用所有特征） ===")
    # # 合并训练集和验证集用于最终模型训练
    # X_train_full = np.concatenate((X_train, X_val), axis=0)
    # y_binary_train_full = np.concatenate((y_binary_train, y_binary_val), axis=0)
    
    # models = train_models(X_train_full, y_binary_train_full, tune_rf=True)

    # print("\n=== 生成预测结果 ===")
    # train_predictions = get_predictions(models, X_train_full)
    # val_predictions = get_predictions(models, X_val)
    # test_predictions = get_predictions(models, X_test)

    # # 合并 survival 结构用于在训练集上计算 C-index（如果需要）
    # try:
    #     y_surv_train_full = np.concatenate((y_surv_train, y_surv_val))
    # except Exception:
    #     y_surv_train_full = None

    # # 打印评估指标（与 1.py 类似）
    # evaluate_and_print_metrics(train_predictions, y_binary_train_full, '训练集', y_surv=y_surv_train_full, compute_ci=False)
    # evaluate_and_print_metrics(val_predictions, y_binary_val, '验证集', y_surv=y_surv_val, compute_ci=False)
    # evaluate_and_print_metrics(test_predictions, y_binary_test, '测试集', y_surv=y_surv_test, compute_ci=True, n_boot=1000)

    # print("\n=== 绘制ROC曲线对比 ===")
    # plot_roc_curves(train_predictions, test_predictions, y_binary_train_full, y_binary_test, 
    #                 save_name="roc_comparison_traditional_models", plot_train=True)
