# reporting.py
import pandas as pd
from textwrap import indent


def print_section(title: str):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def print_subsection(title: str):
    print(f"\n> {title}")


def summarize_model_results(results):
    if not results:
        return

    df = pd.DataFrame(results)
    print_section("Summary of models (macro Fβ=2)")
    # Order columns if present
    cols = [c for c in ["name", "features", "var_quantile",
                        "Fbeta_train", "Fbeta_val", "Fbeta_test"]
            if c in df.columns]
    df = df[cols]
    print(df.to_string(index=False))


def pretty_print_thresholds(thresholds, top_k=2):
    print_section("Simple gene-based rules")

    # Healthy vs Sick
    hs_sorted = sorted(
        thresholds.items(),
        key=lambda item: item[1]["healthy_vs_sick"]["balanced_acc"],
        reverse=True,
    )[:top_k]

    print_subsection(f"Top {top_k} genes – Healthy vs. Sick")
    for gene, info in hs_sorted:
        hs = info["healthy_vs_sick"]
        rule = (
            f"If {gene} expression ≥ {hs['threshold']:.3f}, classify patient as {hs['positive']}; "
            f"otherwise classify patient as {hs['negative']}."
        )
        details = (
            f"Balanced accuracy: {hs['balanced_acc']:.3f}\n"
            f"Mean Healthy: {hs['mean_healthy']:.3f}, "
            f"Mean Sick: {hs['mean_sick']:.3f}"
        )
        print(f"\n• {gene}")
        print(indent(rule, "  "))
        print(indent(details, "  "))

    # RA vs SLE
    rs_sorted = sorted(
        thresholds.items(),
        key=lambda item: item[1]["ra_vs_sle"]["balanced_acc"],
        reverse=True,
    )[:top_k]

    print_subsection(f"Top {top_k} genes – RA vs. SLE")
    for gene, info in rs_sorted:
        rs = info["ra_vs_sle"]
        if rs["threshold"] is None:
            continue
        rule = (
            f"If {gene} expression ≥ {rs['threshold']:.3f}, classify as {rs['positive']}; "
            f"otherwise classify as {rs['negative']}."
        )
        details = (
            f"Balanced accuracy: {rs['balanced_acc']:.3f}\n"
            f"Mean RA: {rs['mean_ra']:.3f}, "
            f"Mean SLE: {rs['mean_sle']:.3f}"
        )
        print(f"\n• {gene}")
        print(indent(rule, "  "))
        print(indent(details, "  "))
