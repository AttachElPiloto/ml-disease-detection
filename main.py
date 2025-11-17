import numpy as np
import pandas as pd
from loading_data import get_training_and_test_data
from visualize_data import show_data_distribution
from preprocessing import (
    report_zero_fraction,
    plot_gene_variance,
    plot_log_transform,
    prepare_features,
)
from models import (
    run_decision_tree_baseline,
    run_decision_tree_regularized,
    run_xgboost,
    run_xgboost_class_weighted,
)
from autoencoder import load_latent_pipeline, run_xgb_on_latent, save_latent_pipeline
from analysis import (
    plot_gene_importance,
    plot_top_gene_distributions,
    significant_thresholds,
)
from reporting import (
    print_section,
    summarize_model_results,
    pretty_print_thresholds,
)

def evaluate_model(
    raw_patient_series,
    save_dir="saved_pipeline",
    device="cpu",
    verbose=True,
):
    ae, xgb, meta = load_latent_pipeline(save_dir, device=device)
    le         = meta["label_encoder"]
    mask       = meta["mask"]          
    scaler     = meta["scaler"]        
    gene_order = meta["gene_order"]

    x_raw_aligned = raw_patient_series.reindex(gene_order)

    x_raw_aligned = x_raw_aligned.fillna(0.0).astype(float)

    x_log = np.log1p(x_raw_aligned.values).reshape(1, -1)

    x_filt = x_log[:, mask]  

    x_z = scaler.transform(x_filt)  

    from autoencoder import encode

    Z = encode(ae, x_z, device=device) 


    proba = xgb.predict_proba(Z)[0]
    class_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([class_idx])[0]
    classes = le.classes_

    if verbose:
        print("=== AE + XGB prediction for one patient ===")
        print(f"Predicted label: {pred_label}")
        print("Class probabilities:")
        for cls, p in zip(classes, proba):
            print(f"  {cls:8s}: {p:.3f}")
    return pred_label, proba, classes

def analyzis():
    # 1 Data loading
    print_section("Data loading")
    healthy_tr, ra_tr, sle_tr, training_data, test_data, y_tr, y_te = get_training_and_test_data()

    # 2 Data visualization and preprocessing
    print_section("Exploratory data analysis")
    show_data_distribution(ra_tr, sle_tr, healthy_tr)
    report_zero_fraction(ra_tr)
    plot_gene_variance(training_data)
    plot_log_transform(training_data)

    # 3 ID3 run
    print_section("ID3 Run")
    Xtr_tree, Xte_tree, _, _ = prepare_features(training_data, test_data, var_quantile=0.5)

    results = []

    dt_base, met_dt_base = run_decision_tree_baseline(Xtr_tree, y_tr, Xte_tree, y_te)
    results.append(met_dt_base)

    dt_reg, met_dt_reg = run_decision_tree_regularized(Xtr_tree, y_tr, Xte_tree, y_te)
    results.append(met_dt_reg)

    # 4 XGBoost run
    print_section("XGBoost on high-variance genes")
    Xtr_xgb, Xte_xgb, mask_xgb, scaler = prepare_features(training_data, test_data, var_quantile=0.96)
    xgb_model, le, y_tr_enc, y_te_enc, met_xgb = run_xgboost(
        Xtr_xgb, y_tr, Xte_xgb, y_te, var_quantile_used=0.96
    )
    results.append(met_xgb)

    class_weight = {"Healthy": 1.0, "RA": 1.0, "SLE": 1.5}
    xgb_weighted, le_w, _, _, met_xgb_w = run_xgboost_class_weighted(
        Xtr_xgb, y_tr, Xte_xgb, y_te, class_weight=class_weight
    )
    met_xgb_w["name"] = "XGBoost_weighted"
    results.append(met_xgb_w)

    # # 5 Important genes detection and thresholds printing
    print_section("Important genes and simple rules")
    gene_names = training_data.columns[mask_xgb]
    imp_df, top_genes = plot_gene_importance(xgb_weighted, gene_names, top_k=20)
    plot_top_gene_distributions(training_data, y_tr, top_genes)

    thresholds = significant_thresholds(training_data, y_tr, top_genes)
    pretty_print_thresholds(thresholds, top_k=2)

    # 6 Autoencoder + XGBoost
    print_section("Autoencoder + XGBoost on latent space")
    ae, xgb_latent, le_latent, met_latent = run_xgb_on_latent(
        Xtr_xgb, Xte_xgb, y_tr, y_te, latent_dim=64
    )
    results.append(met_latent)

    # Global summary
    summarize_model_results(results)

def main():
    # analyzis()
    
    print("SINGLE PATIENT TEST")

    ra_df      = pd.read_csv("./test_data/ra_test_data.tsv", sep="\t", index_col=0)
    sle_df     = pd.read_csv("./test_data/sle_test_data.tsv", sep="\t", index_col=0)
    healthy_df = pd.read_csv("./test_data/healthy_test_data.tsv", sep="\t", index_col=0)

    ra_patient      = ra_df.iloc[15]
    sle_patient     = sle_df.iloc[56]
    healthy_patient = healthy_df.iloc[13]

    print("--------------------------------")
    print(" 1: Predicting RA patient:")
    evaluate_model(ra_patient, save_dir="saved_pipeline", device="cpu", verbose=True)
    print("--------------------------------")
    print(" 2: Predicting SLE patient:")
    evaluate_model(sle_patient, save_dir="saved_pipeline", device="cpu", verbose=True)
    print("--------------------------------")
    print(" 3: Predicting Healthy patient:")
    evaluate_model(healthy_patient, save_dir="saved_pipeline", device="cpu", verbose=True)

if __name__ == "__main__":
    main()
