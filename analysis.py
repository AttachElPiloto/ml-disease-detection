import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gene_importance(xgb_model, gene_names, top_k=20):
    importance = xgb_model.get_booster().get_score(importance_type="gain")
    mapping = {
        gene_names[int(k[1:])]: v for k, v in importance.items()
    }

    imp_df = pd.DataFrame(
        {"Gene": mapping.keys(), "Importance": mapping.values()}
    ).sort_values(by="Importance", ascending=False)

    top_genes = imp_df.head(top_k)

    plt.figure(figsize=(8, 6))
    plt.barh(top_genes["Gene"], top_genes["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (Gain)")
    plt.title(f"Top {top_k} Most Important Genes")
    plt.tight_layout()
    plt.show()

    return imp_df, top_genes


def plot_top_gene_distributions(training_data, y_tr, top_genes):
    top_gene_names = top_genes["Gene"].values
    subset = training_data[top_gene_names].copy()
    subset["label"] = y_tr.values

    melted = subset.melt(id_vars="label", var_name="Gene", value_name="Expression")

    plt.figure(figsize=(16, 6))
    sns.boxplot(
        data=melted,
        x="Gene",
        y="Expression",
        hue="label",
        showfliers=False,
    )
    plt.xticks(rotation=75)
    plt.yscale("log")
    plt.title("Value of Top Important Genes (Log Scale)")
    plt.tight_layout()
    plt.show()

    mean_expr = melted.groupby(["Gene", "label"])["Expression"].mean().unstack()
    sns.heatmap(mean_expr, cmap="RdYlBu_r", center=0)
    plt.title("Mean Gene Expression")
    plt.tight_layout()
    plt.show()


def find_best_threshold_balanced_acc(values, labels, positive_label, negative_label):
    """
    Find a threshold τ on `values` that best separates positive_label vs negative_label
    using balanced accuracy = 0.5 * (recall + specificity).
    """
    mask = np.isin(labels, [positive_label, negative_label])
    vals = values[mask]
    labs = labels[mask]

    y_true = (labs == positive_label).astype(int)

    thr_candidates = np.percentile(vals, np.linspace(5, 95, 50))

    best_thr = None
    best_score = -1.0

    for thr in thr_candidates:
        y_pred = (vals >= thr).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        recall = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        score = 0.5 * (recall + spec)

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score


def find_best_threshold_balanced_acc(values, labels, positive_label, negative_label):
    """
    Find a threshold τ on `values` that best separates positive_label vs negative_label
    using balanced accuracy = 0.5 * (recall + specificity).
    """
    mask = np.isin(labels, [positive_label, negative_label])
    vals = values[mask]
    labs = labels[mask]

    y_true = (labs == positive_label).astype(int)

    thr_candidates = np.percentile(vals, np.linspace(5, 95, 50))

    best_thr = None
    best_score = -1.0

    for thr in thr_candidates:
        y_pred = (vals >= thr).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        recall = tp / (tp + fn + 1e-8)
        spec   = tn / (tn + fp + 1e-8)
        score  = 0.5 * (recall + spec)

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score


def significant_thresholds(training_data, y_tr, top_genes):
    top_gene_names = top_genes["Gene"].values
    df = training_data[top_gene_names].copy()
    df["label"] = y_tr.values

    melted = df.melt(id_vars="label", var_name="Gene", value_name="Expression")

    thresholds = {}

    for gene in top_gene_names:
        g = melted[melted["Gene"] == gene].copy()

        # Healthy vs sick
        labels_bin = np.where(g["label"] == "Healthy", "Healthy", "Sick")
        g["bin_label"] = labels_bin

        mean_healthy = g.loc[g["bin_label"] == "Healthy", "Expression"].mean()
        mean_sick    = g.loc[g["bin_label"] == "Sick", "Expression"].mean()

        if mean_healthy >= mean_sick:
            hs_pos, hs_neg = "Healthy", "Sick"
        else:
            hs_pos, hs_neg = "Sick", "Healthy"

        hs_thr, hs_score = find_best_threshold_balanced_acc(
            g["Expression"].values,
            g["bin_label"].values,
            positive_label=hs_pos,
            negative_label=hs_neg,
        )

        # RA vs SLE
        g_rs = g[g["label"].isin(["RA", "SLE"])].copy()

        mean_ra  = g_rs.loc[g_rs["label"] == "RA",  "Expression"].mean()
        mean_sle = g_rs.loc[g_rs["label"] == "SLE", "Expression"].mean()

        if mean_ra >= mean_sle:
            rs_pos, rs_neg = "RA", "SLE"
        else:
            rs_pos, rs_neg = "SLE", "RA"

        rs_thr, rs_score = find_best_threshold_balanced_acc(
            g_rs["Expression"].values,
            g_rs["label"].values,
            positive_label=rs_pos,
            negative_label=rs_neg,
        )

        thresholds[gene] = {
            "healthy_vs_sick": {
                "positive": hs_pos,
                "negative": hs_neg,
                "threshold": hs_thr,
                "balanced_acc": hs_score,
                "mean_healthy": float(mean_healthy),
                "mean_sick": float(mean_sick),
            },
            "ra_vs_sle": {
                "positive": rs_pos,
                "negative": rs_neg,
                "threshold": rs_thr,
                "balanced_acc": rs_score,
                "mean_ra": float(mean_ra) if g_rs.size else np.nan,
                "mean_sle": float(mean_sle) if g_rs.size else np.nan,
            },
        }

    return thresholds


def gene_threshold_print(training_data, y_tr, top_genes):
    thresholds = significant_thresholds(training_data, y_tr, top_genes)
    # Top 2 genes for Healthy vs Sick
    hs_sorted = sorted(
        thresholds.items(),
        key=lambda item: item[1]["healthy_vs_sick"]["balanced_acc"],
        reverse=True,
    )
    top2_hs = hs_sorted[:2]

    print("=== TOP GENES: HEALTHY vs SICK ===\n")
    for gene, info in top2_hs:
        hs = info["healthy_vs_sick"]
        print(f"Gene: {gene}")
        print(
            f"If expression ≥ {hs['threshold']:.3f}, patient must be {hs['positive']}, "
            f"else patient must be {hs['negative']} "
            f"(Balanced accuracy = {hs['balanced_acc']:.3f})\n"
        )

    # Top 2 genes for RA vs SLE
    rs_sorted = sorted(
        thresholds.items(),
        key=lambda item: item[1]["ra_vs_sle"]["balanced_acc"],
        reverse=True,
    )
    top2_rs = rs_sorted[:2]

    print("=== TOP GENES: RA vs SLE ===\n")
    for gene, info in top2_rs:
        rs = info["ra_vs_sle"]
        if rs["positive"] is None:
            continue
        print(f"Gene: {gene}")
        print(
            f"If expression ≥ {rs['threshold']:.3f}, patient must be {rs['positive']}, "
            f"else patient must be {rs['negative']} "
            f"(Balanced accuracy = {rs['balanced_acc']:.3f})\n"
        )
