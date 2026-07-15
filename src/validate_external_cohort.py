# %%
# 原始数据
import time
import pandas as pd
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import warnings
from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.metrics import (
    accuracy_score,
    classification_report,  # noqa: F401
    confusion_matrix,
    roc_curve,
    auc,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.tree import DecisionTreeClassifier  # noqa: F401
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import xgboost as xgb  # noqa: F401
import joblib
import shap  # noqa: F401
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from sklearn.ensemble import AdaBoostClassifier  # noqa: F401
from lightgbm import LGBMClassifier  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB  # noqa: F401
from sklearn.tree import ExtraTreeClassifier  # noqa: F401
from catboost import CatBoostClassifier, Pool  # noqa: F401
from sklearn.impute import KNNImputer  # noqa: F401

from bootstrap_metrics import write_metric_workbook_with_bootstrap

warnings.filterwarnings("ignore")
# %%
t_start = time.time()
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_RANDOM_STATE = 20260713


# label是每个样本对应的真实标签(0或1)，pred_prob是模型输出的对每个样本的预测概率
def specificityCalc(Labels, Predictions):
    MCM = confusion_matrix(Labels, Predictions)
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]
    Condition_negative = tn_sum + fp_sum
    Specificity = tn_sum / Condition_negative
    return Specificity


def roc_auc(y_test, y_pred_proba, modle_name):
    fig_title = (  # noqa: F841
        modle_name + "Receiver Operating Characteristic (ROC) Curve"
    )
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    # plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title(fig_title)
    # plt.legend(loc="lower right")
    # plt.show()
    return roc_auc


df_verification = pd.read_excel("obesity_NAFLD-外部验证353-0816插值.xlsx")


veri = df_verification.copy()
selected_cols = veri[["group", "gender"]]
selected_cols = pd.get_dummies(selected_cols, drop_first=False, dtype=int)
veri = veri.drop(columns=["group", "gender"])
veri = pd.concat([selected_cols, veri], axis=1)
veri.reset_index(drop=True, inplace=True)
X_veri = veri.drop(["group"], axis=1)
y_veri = veri["group"]
X_veri["gender"] = X_veri["gender"].astype(int)


scaler = StandardScaler()  # 标准化
X_veri_std = np.array(X_veri)
X_veri_std = np.insert(
    scaler.fit_transform(X_veri_std[:, 1:]), 0, X_veri_std[:, 0], axis=1
)
# %%
k_splits = 10
n_model = 11
acc_rate = np.zeros([n_model, k_splits + 2 + 2])
model_auc = np.zeros([n_model, k_splits + 2 + 2])
recall = np.zeros([n_model, k_splits + 2 + 2])
fone = np.zeros([n_model, k_splits + 2 + 2])
kappa = np.zeros([n_model, k_splits + 2 + 2])
spec = np.zeros([n_model, k_splits + 2 + 2])


# %%
def stacking_predict(
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    X_verify,
    y_verify,
    meta_clf,
    k_splits,
    X_verify_std,
    categorical_features=["gender"],
):
    model_name = [
        # "CatBoostClassifier",
        "XGBClassifier",
        "SVC",
        "MLPClassifier",
        "KNeighborsClassifier",
        "RandomForestClassifier",
        "AdaBoostClassifier",
        "LGBMClassifier",
        "LogisticRegression",
        "GaussianNB",
        "ExtraTreeClassifier",
    ]
    # y_pred_pop = np.zeros([len(X_verify)*k_splits, 11])
    y_vall = np.zeros([len(model_name), k_splits, len(X_verify)])
    y_v_po = np.zeros([len(model_name), k_splits, len(X_verify)])
    y_v_predict = np.zeros([len(X_verify), len(model_name)])
    for i in range(len(model_name)):
        for j in range(k_splits):

            pkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_allmodel"
            clf_name = model_name[i] + str(j) + ".pkl"
            current_clf = joblib.load(os.path.join(pkl_path, clf_name))
            if (
                isinstance(current_clf, KNeighborsClassifier)
                or isinstance(current_clf, MLPClassifier)
                or isinstance(current_clf, svm.SVC)
                or isinstance(current_clf, LogisticRegression)
            ):
                y_vall[i, j, :] = current_clf.predict(X_verify_std)
                y_v_po[i, j, :] = current_clf.predict_proba(X_verify_std)[:, 1]
            elif isinstance(current_clf, CatBoostClassifier):
                X_verify[categorical_features] = X_verify[categorical_features].astype(
                    int
                )
                test_pool = Pool(X_verify, y_verify, cat_features=categorical_features)
                y_vall[i, j, :] = current_clf.predict(test_pool)
                y_v_po[i, j, :] = current_clf.predict_proba(test_pool)[:, 1]
            else:
                y_vall[i, j, :] = current_clf.predict(X_verify)
                y_v_po[i, j, :] = current_clf.predict_proba(X_verify)[:, 1]

            acc_rate[i, j] = accuracy_score(y_verify, y_vall[i, j, :])

            model_auc[i, j] = roc_auc(y_verify, y_v_po[i, j, :], model_name[i])
            recall[i, j] = recall_score(y_verify, y_vall[i, j, :])
            fone[i, j] = f1_score(y_verify, y_vall[i, j, :])
            kappa[i, j] = cohen_kappa_score(y_verify, y_vall[i, j, :])
            spec[i, j] = specificityCalc(y_verify, y_vall[i, j, :])
    # 投票
    for i in range(len(model_name)):
        for j in range(len(X_verify)):
            if sum(y_vall[i, :, j]) >= k_splits / 2:
                y_v_predict[j, i] = 1
            else:
                y_v_predict[j, i] = 0
    categorical_features = [
        "XGB",
        "SVM",
        "MLP",
        "KNN",
        "RF",
        "ADB",
        "LGBM",
        "LR",
        "NB",
        "ET",
    ]
    Xverify_2 = pd.concat(
        [X_verify, pd.DataFrame(y_v_predict.astype(int), columns=categorical_features)],
        axis=1,
    )

    categorical_features.append("gender")
    # Xverify_2 = pd.DataFrame(y_v_predict, columns=categorical_features)

    Xverify_2[categorical_features] = Xverify_2[categorical_features].astype(int)
    pool = Pool(Xverify_2, y_verify, cat_features=categorical_features)
    y_pred = meta_clf.predict(pool)
    y_pred_proba = meta_clf.predict_proba(pool)[:, 1]
    return acc_rate, model_auc, recall, fone, kappa, spec, y_pred, y_pred_proba, y_v_po


# %%
model_name = "CatBoostClassifier"
pkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_allmodel"
y_pred_probacat = []
for i in range(k_splits):

    clf_name = model_name + str(i) + ".pkl"
    meta_clf = joblib.load(os.path.join(pkl_path, clf_name))
    acc_rate, model_auc, recall, fone, kappa, spec, y_pred_veri, y_pred_proba_veri, base_fold_proba = (
        stacking_predict(
            acc_rate,
            model_auc,
            recall,
            fone,
            kappa,
            spec,
            X_veri,
            y_veri,
            meta_clf,
            k_splits,
            X_veri_std,
        )
    )
    y_pred_probacat = y_pred_probacat + list(y_pred_proba_veri)
    model_auc[n_model - 1, i] = roc_auc(y_veri, y_pred_proba_veri, "stack")
    acc_rate[n_model - 1, i] = accuracy_score(
        y_veri,
        y_pred_veri,
    )
    recall[n_model - 1, i] = recall_score(
        y_veri,
        y_pred_veri,
    )
    fone[n_model - 1, i] = f1_score(
        y_veri,
        y_pred_veri,
    )
    kappa[n_model - 1, i] = cohen_kappa_score(
        y_veri,
        y_pred_veri,
    )
    spec[n_model - 1, i] = specificityCalc(
        y_veri,
        y_pred_veri,
    )
y_pred_probacat = np.array(y_pred_probacat)
stacking_probability_by_model = y_pred_probacat.reshape(k_splits, len(y_veri))
# One external-validation probability per patient is obtained with the median
# across the 10 fold-specific models. The same patient-level probabilities are
# used for both the point estimates and the bootstrap confidence intervals.
stacking_median_probability = np.median(stacking_probability_by_model, axis=0)
base_median_probability = np.median(base_fold_proba, axis=1).T
external_probability_matrix = np.column_stack(
    [base_median_probability, stacking_median_probability]
)
external_prediction_matrix = (external_probability_matrix >= 0.5).astype(int)
df_proba = pd.read_excel('pop_9.1.xlsx')
df_proba ['stacking(catboost)'] = y_pred_probacat 
length = 353
model_fold_list = []
for i in range(10):
    model_fold_list  = model_fold_list + [i+1 for _ in range(length)]
df_proba ['Model_n'] = model_fold_list 

with pd.ExcelWriter("外部验证模型预测概率_9.1.xlsx") as writer:
    df_proba.to_excel(writer,  index=False)
# %%
index_1 = [
    "XGB",
    "SVM",
    "MLP",
    "KNN",
    "RandomForest",
    "ADB",
    "LGBM",
    "LogisticRegression",
    "GaussianNB",
    "ExtraTree",
    "stacking(CatBoost)",
]
fold_metrics_external = {
    "Accuracy": acc_rate,
    "AUC": model_auc,
    "Recall": recall,
    "F1": fone,
    "Kappa": kappa,
    "Specificity": spec,
}
write_metric_workbook_with_bootstrap(
    fold_metrics_external,
    index_1,
    y_veri,
    external_prediction_matrix,
    external_probability_matrix,
    "external_validation_results_median_bootstrap_ci.xlsx",
    n_folds=k_splits,
    n_bootstrap=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_RANDOM_STATE,
)

patient_prediction_table = pd.DataFrame(
    external_probability_matrix,
    columns=[f"{label}_median_probability" for label in index_1],
)
patient_prediction_table.insert(0, "outcome", np.asarray(y_veri, dtype=int))
patient_prediction_table.to_excel(
    "external_validation_patient_level_median_probabilities.xlsx",
    index=False,
)
