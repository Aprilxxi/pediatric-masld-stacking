import sys
sys.path.insert(0, r'D:\Python code\mechine_learning_stroke')
# %%
# 原始数据
import time
import pandas as pd
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  
import seaborn as sns  
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.model_selection import KFold
import joblib
import os
from scipy.special import softmax, expit
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import AdaBoostClassifier  
from lightgbm import LGBMClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.tree import ExtraTreeClassifier  
from catboost import CatBoostClassifier, Pool  
from sklearn.impute import KNNImputer  
from bootstrap_metrics import (
    patient_bootstrap_metric_tables,
    write_metric_workbook_with_bootstrap,
)


warnings.filterwarnings("ignore")
# %%
t_start = time.time()
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_RANDOM_STATE = 20260713


# label是每个样本对应的真实标签(0或1)，pred_prob是模型输出的对每个样本的预测概率
# FPR, TPR, _ = roc_curve(label, pred_prob, pos_label=1)
# AUC = roc_auc_score(label, pred_prob)





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


best_fold = []


# %%
def train_base(
    clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
):
    score = 0  # noqa: F841
    auc_ave = 0
    acc_ave = 0
    i = 0  # noqa: F841
    best_f = -1
    model_name = str(type(clf)).split(".")[-1][0:-2]
    for train_index, test_index in kf.split(y):
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]  # noqa: F841
        clf.fit(X_train, y_train)
        y_pre[test_index, model_n] = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        y_pre_p[test_index, model_n] = y_proba[:, 1]  # 测试集判定为1的概率
        acc_rate[model_n, i] = accuracy_score(y_test, y_pre[test_index, model_n])
        acc_ave = acc_ave + acc_rate[model_n, i]
        model_auc[model_n, i] = roc_auc(
            y_test, y_pre_p[test_index, model_n], model_name
        )
        auc_ave = auc_ave + model_auc[model_n, i]
        recall[model_n, i] = recall_score(y_test, y_pre[test_index, model_n])
        fone[model_n, i] = f1_score(y_test, y_pre[test_index, model_n])
        kappa[model_n, i] = cohen_kappa_score(y_test, y_pre[test_index, model_n])
        spec[model_n, i] = specificityCalc(y_test, y_pre[test_index, model_n])
        print(f"Accuracy score : {acc_rate[model_n, i]}")
        # break
        if model_auc[model_n, i] > score:
            best_clf = clf  # noqa: F841
            score = model_auc[model_n, i]
            best_f = i
        pkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_allmodel"
        clf_name = model_name + str(i) + ".pkl"
        i += 1
        joblib.dump(clf, os.path.join(pkl_path, clf_name))
        # print(classification_report(y_pre[test_index, model_n], y_test))
    bpkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_bestmodel"
    best_name = model_name + ".pkl"
    best_fold.append(best_f)
    joblib.dump(best_clf, os.path.join(bpkl_path, best_name))
    print("平均acc", acc_ave / 10)
    print("平均auc", auc_ave / 10)
    return y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold


# %%
df = pd.read_excel("obesity-5936-0716.xlsx")
df_1555 = pd.read_excel("obesity-1555-0815.xlsx")
df_verification = pd.read_excel("External verification-354-0815.xlsx")

df_1555 = df_1555.drop(columns="ID")
df_verification = df_verification.drop(columns="ID")

drop_features = [
    "Large platelet ratio",
    "Red blood cell distribution width sd",
    "Creatine kinase isoenzyme",
    "chlorine",
    "Cystatin _c",
    "potassium",
    "sodium",
    "phosphorus",
    "Retinol binding protein",
]
df_1555 = df_1555.drop(columns=drop_features)
df = df.drop(columns=drop_features)
df_verification = df_verification.drop(columns=drop_features)

drop_features = ["black", "tyg", "tg/hdl", "tc/hdl", "nohdl", "lhr", "homa-ir"]
# drop_features  = ['Absolute value of lymphocytes','Glucose','High-density lipoprotein cholesterol','Total cholesterol','Triacylglycerol','Insulin']
df = df.drop(columns=drop_features)
df = pd.concat([df, df_1555])
# %%
index_names = df[df["bmiz"] < 1].index
df.drop(index_names, inplace=True)
df = df.reset_index(drop=True)
index_names = df_verification[df_verification["bmiz"] < 1].index
df_verification.drop(index_names, inplace=True)
df_verification = df_verification.reset_index(drop=True)
# %%
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
# %%
# for col in numeric_cols:
#     plt.figure(figsize=(8, 6))
#     sns.displot(data=df[col], kde=True, color="orange")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.title(f"Histogram of {col}")
#     plt.show()

# %%
# for col in numeric_cols:
#     if col == "age" or col == "gender" or col == "group":
#         continue
#     plt.figure(figsize=(8, 6))
#     plt.title(f"scatterplot of {col}")
#     sns.scatterplot(data=df, x="age", y=col, hue="gender")

# %%
veri = df.copy()
selected_cols = veri[["group", "gender"]]
selected_cols = pd.get_dummies(selected_cols, drop_first=False, dtype=int)
veri = df.drop(columns=["group", "gender"])
veri = pd.concat([selected_cols, veri], axis=1)
veri.reset_index(drop=True, inplace=True)
X = veri.drop(["group"], axis=1)
y = veri["group"]
X["gender"] = X["gender"].astype(int)


# %%


scaler = StandardScaler()
scaler.fit(X)
X_std = np.array(X)
joblib.dump(scaler, os.path.join('D:/Python code/mechine_learning_stroke/NAFLD_allmodel', 'scaler.pkl' ))
X_std = np.insert(scaler.fit_transform(X_std[:, 1:]), 0, X_std[:, 0], axis=1)

imputer = KNNImputer(n_neighbors=80)
X_imputer = imputer.fit_transform(X_std, y)
X_imputer = np.insert(
    scaler.inverse_transform(X_imputer[:, 1:]), 0, X_imputer[:, 0], axis=1
)


X_imputer = pd.DataFrame(X_imputer, columns=X.columns)
X = X_imputer
# %% 插值写入excel
df_imputer = pd.concat([y, X_imputer], axis=1)
with pd.ExcelWriter("obesity_NAFLD-5936+1555-0816插值.xlsx") as writer:
    df_imputer.to_excel(writer, index=False)
# with pd.ExcelWriter('obesity-5936-0716_融合指标版插值.xlsx') as writer:
#     df_imputer.to_excel(writer, index=False)


# %%%
veri = df_verification.copy()
selected_cols = veri[["group", "gender"]]
selected_cols = pd.get_dummies(selected_cols, drop_first=False, dtype=int)
veri = veri.drop(columns=["group", "gender"])
veri = pd.concat([selected_cols, veri], axis=1)
veri.reset_index(drop=True, inplace=True)
X_veri = veri.drop(["group"], axis=1)
y_veri = veri["group"]
X_veri["gender"] = X_veri["gender"].astype(int)
scaler = StandardScaler()
scaler.fit(X_veri)
X_veri_std = np.array(X_veri)
X_veri_std = np.insert(
    scaler.fit_transform(X_veri_std[:, 1:]), 0, X_veri_std[:, 0], axis=1
)


X_imputer = imputer.fit_transform(X_veri_std, y_veri)
X_imputer = np.insert(
    scaler.inverse_transform(X_imputer[:, 1:]), 0, X_imputer[:, 0], axis=1
)


X_imputer = pd.DataFrame(X_imputer, columns=X_veri.columns)
X_veri = X_imputer

df_imputer = pd.concat([y_veri, X_imputer], axis=1)
with pd.ExcelWriter("obesity_NAFLD-外部验证353-0816插值.xlsx") as writer:
    df_imputer.to_excel(writer, index=False)

# %%
scaler = StandardScaler()  # 标准化
X_std = np.array(X)
X_std = np.insert(scaler.fit_transform(X_std[:, 1:]), 0, X_std[:, 0], axis=1)
scaler = StandardScaler()  # 标准化
X_veri_std = np.array(X_veri)
X_veri_std = np.insert(scaler.fit_transform(X_veri_std[:, 1:]), 0, X_veri_std[:, 0], axis=1)


# %%
k_splits = 10
kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
n_model = 11
y_pre = np.zeros([len(X), n_model])
y_pre_p = np.zeros([len(X), n_model])
acc_rate = np.zeros([n_model + 1, k_splits + 2+2])
model_auc = np.zeros([n_model + 1, k_splits + 2+2])
recall = np.zeros([n_model + 1, k_splits + 2+2])
fone = np.zeros([n_model + 1, k_splits + 2+2])
kappa = np.zeros([n_model + 1, k_splits + 2+2])
spec = np.zeros([n_model + 1, k_splits + 2+2])


# %% model_n = 0

dt_clf = DecisionTreeClassifier(max_depth=5)
model_n = 0
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spe, best_fold = train_base(
    dt_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)
# %%  XGB model_n = 1
xgb_clf = xgb.XGBClassifier(
    learning_rate=0.008,  # 学习率
    n_estimators=800,  # 树的数量
    max_depth=5,  # 树的最大深度
    subsample=0.75,  # 子样本比例
    colsample_bytree=0.8,  # 每棵树使用的特征比例
    gamma=0,  # 节点分裂所需的最小损失减少值
    reg_alpha=5,  # L1 正则化系数
    reg_lambda=200,  # L2 正则化系数
)
model_n = 1
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    xgb_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)

# %%   SVM model_n = 2
svm_clf = svm.SVC(
    kernel="rbf",  # 核函数 linear、poly rbf sigmoid
    C=0.8,  # 0-1
    gamma=0.002,  # 默认为样本特征数量的倒数
    probability=True,
)
model_n = 2
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    svm_clf,
    kf,
    X_std,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)
# %% MLP model_n = 3
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(55,),  # 指定每个隐藏层的神经元数量
    max_iter=1000,  # 最大迭代次数
    alpha=0.1,  # L2正则化参数，防止过拟合
    solver="adam",  # 权重优化算法
    random_state=42,  # 种子
    learning_rate_init=0.08,
)
model_n = 3
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    mlp_clf,
    kf,
    X_std,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)

# %% KNN model_n = 4


knn_clf = KNeighborsClassifier(
    n_neighbors=180,
    weights="uniform",  # str或callable，可选(默认='uniform')
    algorithm="auto",
    p=1,  # p=2默认为欧拉距离，p=1为曼哈顿距离，float('inf')切比雪夫距离
)
model_n = 4

y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    knn_clf,
    kf,
    X_std,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)


# %% model_n = 5


rf_clf = RandomForestClassifier(
    max_depth=11,  # 树的最大深度
    min_samples_leaf=5,  # 在叶节点处需要的最小样本数
    min_samples_split=10,  # 拆分内部节点所需的最少样本数
    n_estimators=485,  # 森林中树木的数量
    random_state=42,
    n_jobs=-1,  # 要并行运行的作业的数量
)
model_n = 5
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    rf_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)

# %% model_n = 6
# ADB


base_classifier = DecisionTreeClassifier(max_depth=3)
ada_clf = AdaBoostClassifier(
    estimator=base_classifier,
    n_estimators=200,  # 200
    learning_rate=0.3,  # 0.4
    random_state=42,
)
model_n = 6
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    ada_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)

# %%
model_name = "AdaBoostClassifier"
j = 0
# %% model_n = 7
# 训练模型
lgbm_clf = LGBMClassifier(
    feature_fraction=0.6,
    learning_rate=0.005,
    max_depth=5,
    num_leaves=2 ^ 4, # 2 ^max_depth-1
    subsample=0.8,
    verbosity=-1,
    n_estimators=800,
    min_child_weight= 1,
    lambda_l1=5,
    lambda_l2 = 200,
)

model_n = 7
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    lgbm_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)
# %%
model_name = "LGBMClassifier"
j = 0

# %% Logistic regression model_n = 8


LR_reg = LogisticRegression(
    #class_weight="balanced",
    penalty="l1",  # 正则化类型
    C=0.02,  # 正则化强度
    random_state=42,
    solver="liblinear",  # 损失函数优化
    multi_class="auto",
)

model_n = 8
y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    LR_reg,
    kf,
    X_std,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)
# %%
model_name = "LogisticRegression"
j = 0

# %% 朴素贝叶斯 model_n = 9



nb_clf = GaussianNB(var_smoothing = 1e-5)
model_n = 9

y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    nb_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)


# %% model_n = 10


et_clf = ExtraTreeClassifier(
    # class_weight="balanced",
    # n_estimators = 200,
    criterion="log_loss",
    max_features=20,
    random_state=42,
    max_depth=10,  # 树的最大深度
    min_samples_leaf=20,  # 在叶节点处需要的最小样本数
    min_samples_split=50,  # 拆分内部节点所需的最少样本数
)
model_n = 10


y_pre, y_pre_p, acc_rate, model_auc, recall, fone, kappa, spec, best_fold = train_base(
    et_clf,
    kf,
    X,
    y,
    model_n,
    y_pre,
    y_pre_p,
    acc_rate,
    model_auc,
    recall,
    fone,
    kappa,
    spec,
    best_fold,
)


# %% catboost



categorical_features = [
    # "DT",
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

X_2 = pd.concat(
    [X, pd.DataFrame(y_pre[:, 1:].astype(int), columns=categorical_features)], axis=1
)

categorical_features.append("gender")
# X_2 = pd.DataFrame(y_pre[:,1:].astype(int), columns=categorical_features)
X_2[categorical_features] = X_2[categorical_features].astype(int)
# %%
catclf = CatBoostClassifier(
    # auto_class_weights="Balanced",
    iterations=1250,
    learning_rate=0.004,
    depth=4,
    loss_function="Logloss",  # 损失函数 二分类任务使用'Logloss' 概率用 CrossEntropy
    verbose=250,  # 每100次迭代打印一次信息
    l2_leaf_reg=300,
    eval_metric="Accuracy",
    subsample=0.75,
    grow_policy="Depthwise",  # Depthwise（整层生长，同xgb）、Lossguide（叶子结点生长，同lgb）
    min_data_in_leaf=50,
    leaf_estimation_method='Gradient'
)
i = 0
model_n = 11
score = 0
model_name = str(type(catclf)).split(".")[-1][0:-2]

y_pred_pcab = np.zeros([len(X),1])
y_class_pcab = np.zeros([len(X),1], dtype=int)

for train_index, test_index in kf.split(y):
    # if i !=1:
    #     i += 1
    #     continue
    X_train, X_test = X_2.loc[train_index], X_2.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features)
    catclf.fit(train_pool)
    y_pred = catclf.predict(test_pool)
    y_pred_proba = catclf.predict_proba(test_pool)[:, 1]
    y_pred_pcab[test_index,0] = y_pred_proba
    y_class_pcab[test_index,0] = np.asarray(y_pred, dtype=int).reshape(-1)

    acc_rate[model_n, i] = accuracy_score(y_test, y_pred)
    model_auc[model_n, i] = roc_auc(y_test, y_pred_proba, model_name)
    recall[model_n, i] = recall_score(y_test, y_pred)
    fone[model_n, i] = f1_score(y_test, y_pred)
    kappa[model_n, i] = cohen_kappa_score(y_test, y_pred)
    spec[model_n, i] = specificityCalc(y_test, y_pred)
    print(f"Accuracy score : {acc_rate[model_n, i]}")
    print(f"AUC : {model_auc[model_n, i]}")
    # break
    if model_auc[model_n, i] > score:
        best_clf = catclf
        score = model_auc[model_n, i]
    pkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_allmodel"
    clf_name = model_name + str(i) + ".pkl"
    i += 1
    joblib.dump(catclf, os.path.join(pkl_path, clf_name))
bpkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_bestmodel"
best_name = model_name + ".pkl"
joblib.dump(best_clf, os.path.join(bpkl_path, best_name))


# %%
index_1 = [
    "DecisionTree",
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
fold_metrics_oof = {
    "Accuracy": acc_rate,
    "AUC": model_auc,
    "Recall": recall,
    "F1": fone,
    "Kappa": kappa,
    "Specificity": spec,
}
oof_pred_matrix = np.column_stack([y_pre, y_class_pcab[:, 0]])
oof_proba_matrix = np.column_stack([y_pre_p, y_pred_pcab[:, 0]])
write_metric_workbook_with_bootstrap(
    fold_metrics_oof,
    index_1,
    y,
    oof_pred_matrix,
    oof_proba_matrix,
    "training_results_bootstrap_ci.xlsx",
    n_folds=k_splits,
    n_bootstrap=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_RANDOM_STATE,
)

# %%
# df_y_pred = pd.DataFrame(y_pred.T, columns=col)
# df_y_pred_proba = pd.DataFrame(y_pred_proba.T, columns=col)

# y_pred
# y_pred_proba
# %%
print(time.time() - t_start)
# %%
plt.figure(figsize=(20, 12))
plt.barh(X_2.columns, catclf.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("CatBoost Feature Importances")
plt.show()
# %%
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# %%

col_feature = np.array(X.columns)
# %%

pkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_bestmodel"
clf_name = "AdaBoostClassifier.pkl"
clf = joblib.load(os.path.join(pkl_path, clf_name))


feature_imp = pd.DataFrame(np.vstack((col_feature, clf.feature_importances_)).T)

# %%
with pd.ExcelWriter("AdaBoostfeature_importances0.xlsx") as writer:
    feature_imp.to_excel(writer,  index=False)

# %%
model_name = "XGBClassifier"
j = 0
# %%
model_name = "LGBMClassifier"
j = 0
# %%
model_name = "RandomForestClassifier"
j = 0
# %%
model_name = "ExtraTreeClassifier"
j = 0
# %%
model_name = "AdaBoostClassifier"
j = 0
# %%
model_name = "LogisticRegression"
j = 0
 # %%

model_name_all = [


    "SVC",
    "MLPClassifier",
    "KNeighborsClassifier",
    "GaussianNB",

]


# %%
# %% predict_new
def stacking_predict(X_verify, y_verify, meta_clf, k_splits, X_verify_std):
    model_name = [
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
    y_vall = np.zeros([len(model_name), k_splits, len(X_verify)])
    y_vp = np.zeros([len(X_verify), len(model_name)])
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
            else:
                y_vall[i, j, :] = current_clf.predict(X_verify)
    # 投票
    for i in range(len(model_name)):
        for j in range(len(X_verify)):
            if sum(y_vall[i, :, j]) >= k_splits / 2:
                y_vp[j, i] = 1
            else:
                y_vp[j, i] = 0
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
        [X_verify, pd.DataFrame(y_vp.astype(int), columns=categorical_features)], axis=1
    )

    categorical_features.append("gender")

    Xverify_2[categorical_features] = Xverify_2[categorical_features].astype(int)
    pool = Pool(Xverify_2, y_verify, cat_features=categorical_features)
    y_pred = meta_clf.predict(pool)
    y_pred_proba = meta_clf.predict_proba(pool)[:, 1]
    return y_pred, y_pred_proba


# %%
pkl_path = "D:/Python code/mechine_learning_stroke/NAFLD_bestmodel"
clf_name = "CatBoostClassifier.pkl"
meta_clf = joblib.load(os.path.join(pkl_path, clf_name))


y_pred_veri, y_pred_proba_veri = stacking_predict(
    X_veri, y_veri, meta_clf, k_splits, X_veri_std
)
# %%
print('外部验证auc',roc_auc(y_veri, y_pred_proba_veri, 'stack'))
print('外部验证accuracy',accuracy_score(y_veri, y_pred_veri,))
print('外部验证recall/灵敏度',recall_score(y_veri, y_pred_veri,))
print('外部验证f1',f1_score(y_veri, y_pred_veri,))
print('外部验证kappa',cohen_kappa_score(y_veri, y_pred_veri,))
print('外部验证specificity',specificityCalc(y_veri, y_pred_veri,))

external_bootstrap_tables = patient_bootstrap_metric_tables(
    y_veri,
    np.asarray(y_pred_veri, dtype=int).reshape(-1, 1),
    np.asarray(y_pred_proba_veri, dtype=float).reshape(-1, 1),
    ["stacking(CatBoost)"],
    n_bootstrap=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_RANDOM_STATE + 1,
)
with pd.ExcelWriter("external_stacking_results_bootstrap_ci.xlsx") as writer:
    external_summary = []
    for metric_name, metric_table in external_bootstrap_tables.items():
        metric_table.to_excel(writer, sheet_name=metric_name, index=True)
        long_table = metric_table.reset_index(names="model")
        long_table.insert(1, "metric", metric_name)
        external_summary.append(long_table)
    pd.concat(external_summary, ignore_index=True).to_excel(
        writer, sheet_name="Bootstrap summary", index=False
    )
