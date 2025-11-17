import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------- ID3 ----------

def run_decision_tree_baseline(Xtr_z, y_tr, Xte_z, y_te, beta=2.0, plot=True):
    X_train, X_valid, y_train, y_valid = train_test_split(
        Xtr_z, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        random_state=42,
    )
    dt.fit(X_train, y_train)

    train_pred = dt.predict(X_train)
    valid_pred = dt.predict(X_valid)
    test_pred = dt.predict(Xte_z)

    fb_train = fbeta_score(y_train, train_pred, average="macro", beta=beta)
    fb_valid = fbeta_score(y_valid, valid_pred, average="macro", beta=beta)
    fb_test  = fbeta_score(y_te,    test_pred,  average="macro", beta=beta)

    print("=== ID3 ===")
    print(f"Train Fβ={beta} (macro): {fb_train:.3f}")
    print(f"Valid Fβ={beta} (macro): {fb_valid:.3f}")
    print(f"Test  Fβ={beta} (macro): {fb_test:.3f}")
    print()

    if plot:
        _plot_conf_mats(dt.classes_, y_valid, valid_pred, y_te, test_pred,
                        title_prefix="ID3")

    metrics = {
        "name": "DecisionTree_full",
        "features": Xtr_z.shape[1],
        "var_quantile": 0.5,
        "Fbeta_train": float(fb_train),
        "Fbeta_val": float(fb_valid),
        "Fbeta_test": float(fb_test),
    }

    return dt, metrics


def run_decision_tree_regularized(Xtr_z, y_tr, Xte_z, y_te, beta=2.0, plot=True):
    X_train, X_valid, y_train, y_valid = train_test_split(
        Xtr_z, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42,
    )
    dt.fit(X_train, y_train)

    train_pred = dt.predict(X_train)
    valid_pred = dt.predict(X_valid)
    test_pred = dt.predict(Xte_z)

    fb_train = fbeta_score(y_train, train_pred, average="macro", beta=beta)
    fb_valid = fbeta_score(y_valid, valid_pred, average="macro", beta=beta)
    fb_test  = fbeta_score(y_te,    test_pred,  average="macro", beta=beta)

    print("=== ID3 (max_depth=4) ===")
    print(f"Train Fβ={beta} (macro): {fb_train:.3f}")
    print(f"Valid Fβ={beta} (macro): {fb_valid:.3f}")
    print(f"Test  Fβ={beta} (macro): {fb_test:.3f}")
    print()

    if plot:
        _plot_conf_mats(dt.classes_, y_valid, valid_pred, y_te, test_pred,
                        title_prefix="Tree (depth=4)")

    metrics = {
        "name": "DecisionTree_depth4",
        "features": Xtr_z.shape[1],
        "var_quantile": 0.5,
        "Fbeta_train": float(fb_train),
        "Fbeta_val": float(fb_valid),
        "Fbeta_test": float(fb_test),
    }

    return dt, metrics


def _plot_conf_mats(classes, y_valid, valid_pred, y_test, test_pred, title_prefix=""):
    cm_val = confusion_matrix(y_valid, valid_pred, labels=classes)
    cm_test = confusion_matrix(y_test, test_pred, labels=classes)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=classes)
    disp_val.plot(cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title(f"{title_prefix} – Validation")

    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
    disp_test.plot(cmap="Blues", ax=axes[1], colorbar=False)
    axes[1].set_title(f"{title_prefix} – Test")

    plt.tight_layout()
    plt.show()



def run_xgboost(
    Xtr_z,
    y_tr,
    Xte_z,
    y_te,
    beta=2.0,
    var_quantile_used=0.96,
    plot=True,
):
    le = LabelEncoder()
    y_tr_encoded = le.fit_transform(y_tr)
    y_te_encoded = le.transform(y_te)

    X_tr, X_te = Xtr_z, Xte_z

    X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
        X_tr, y_tr_encoded, test_size=0.2, random_state=42, stratify=y_tr_encoded
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        n_estimators=6000,
        learning_rate=0.2,
        subsample=0.04,
        colsample_bytree=0.35,
        reg_lambda=1.0,
        reg_alpha=0.1,
        early_stopping_rounds=100,
        random_state=290,
    )

    xgb.fit(
        X_tr_in,
        y_tr_in,
        eval_set=[(X_val_in, y_val_in)],
        verbose=False,
    )

    y_tr_pred = xgb.predict(X_tr_in)
    y_val_pred = xgb.predict(X_val_in)
    y_te_pred = xgb.predict(X_te)

    fb_train = fbeta_score(y_tr_in,      y_tr_pred,  average="macro", beta=beta)
    fb_val   = fbeta_score(y_val_in,     y_val_pred, average="macro", beta=beta)
    fb_test  = fbeta_score(y_te_encoded, y_te_pred,  average="macro", beta=beta)

    print(f"=== XGBoost (top {int(var_quantile_used*100)}% variance genes) ===")
    print(f"Train Fβ={beta} (macro): {fb_train:.3f}")
    print(f"Val   Fβ={beta} (macro): {fb_val:.3f}")
    print(f"Test  Fβ={beta} (macro): {fb_test:.3f}")
    print()

    if plot:
        cm_val = confusion_matrix(le.inverse_transform(y_val_in),
                                  le.inverse_transform(y_val_pred),
                                  labels=le.classes_)
        cm_test = confusion_matrix(le.inverse_transform(y_te_encoded),
                                   le.inverse_transform(y_te_pred),
                                   labels=le.classes_)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(cm_val, display_labels=le.classes_).plot(
            cmap="Blues", ax=axes[0], colorbar=False
        )
        axes[0].set_title("XGBoost – Validation")
        ConfusionMatrixDisplay(cm_test, display_labels=le.classes_).plot(
            cmap="Blues", ax=axes[1], colorbar=False
        )
        axes[1].set_title("XGBoost – Test")
        plt.tight_layout()
        plt.show()

    metrics = {
        "name": "XGBoost_raw",
        "features": Xtr_z.shape[1],
        "var_quantile": var_quantile_used,
        "Fbeta_train": float(fb_train),
        "Fbeta_val": float(fb_val),
        "Fbeta_test": float(fb_test),
    }

    return xgb, le, y_tr_encoded, y_te_encoded, metrics


def run_xgboost_class_weighted(
    Xtr_z,
    y_tr,
    Xte_z,
    y_te,
    class_weight,
    beta=2.0,
    plot=True,
):
    le = LabelEncoder()
    y_tr_encoded = le.fit_transform(y_tr)
    y_te_encoded = le.transform(y_te)

    X_tr, X_te = Xtr_z, Xte_z

    X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
        X_tr, y_tr_encoded, test_size=0.2, random_state=42, stratify=y_tr_encoded
    )

    label_to_weight = {le.transform([name])[0]: w for name, w in class_weight.items()}

    sw_tr  = np.array([label_to_weight[c] for c in y_tr_in], dtype=float)
    sw_val = np.array([label_to_weight[c] for c in y_val_in], dtype=float)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        n_estimators=6000,
        learning_rate=0.2,
        subsample=0.04,
        colsample_bytree=0.35,
        reg_lambda=1.0,
        reg_alpha=0.1,
        early_stopping_rounds=250,
        random_state=42,
    )

    xgb.fit(
        X_tr_in,
        y_tr_in,
        sample_weight=sw_tr,
        eval_set=[(X_val_in, y_val_in)],
        sample_weight_eval_set=[sw_val],
        verbose=False,
    )

    y_tr_pred = xgb.predict(X_tr_in)
    y_val_pred = xgb.predict(X_val_in)
    y_te_pred  = xgb.predict(X_te)

    fb_train = fbeta_score(y_tr_in,      y_tr_pred,  average="macro", beta=beta)
    fb_val   = fbeta_score(y_val_in,     y_val_pred, average="macro", beta=beta)
    fb_test  = fbeta_score(y_te_encoded, y_te_pred,  average="macro", beta=beta)

    print("=== XGBoost with class weights (early stopping on validation) ===")
    print(f"Train Fβ={beta} (macro): {fb_train:.3f}")
    print(f"Val   Fβ={beta} (macro): {fb_val:.3f}")
    print(f"Test  Fβ={beta} (macro): {fb_test:.3f}")
    print()

    if plot:
        cm_val = confusion_matrix(le.inverse_transform(y_val_in),
                                  le.inverse_transform(y_val_pred),
                                  labels=le.classes_)
        cm_test = confusion_matrix(le.inverse_transform(y_te_encoded),
                                   le.inverse_transform(y_te_pred),
                                   labels=le.classes_)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(cm_val, display_labels=le.classes_).plot(
            cmap="Blues", ax=axes[0], colorbar=False
        )
        axes[0].set_title("XGB weighted – Validation")

        ConfusionMatrixDisplay(cm_test, display_labels=le.classes_).plot(
            cmap="Blues", ax=axes[1], colorbar=False
        )
        axes[1].set_title("XGB weighted – Test")

        plt.tight_layout()
        plt.show()

    metrics = {
        "name": "XGBoost_weighted",
        "features": Xtr_z.shape[1],
        "var_quantile": 0.96,
        "Fbeta_train": float(fb_train),
        "Fbeta_val": float(fb_val),
        "Fbeta_test": float(fb_test),
    }

    return xgb, le, y_tr_encoded, y_te_encoded, metrics
