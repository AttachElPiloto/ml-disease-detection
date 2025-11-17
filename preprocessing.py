import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def report_zero_fraction(ra_tr):
    zero_fraction = (ra_tr.values == 0).sum() / ra_tr.values.size
    print("Fraction of zeros in RA train:", zero_fraction)


def plot_gene_variance(training_data):
    gene_var = training_data.var(axis=0)
    plt.hist(gene_var, bins=50)
    plt.yscale("log")
    plt.title("Gene variance distribution")
    plt.xlabel("Variance")
    plt.ylabel("Count (log)")
    plt.show()


def plot_log_transform(training_data):
    raw = training_data.values.flatten()
    log_raw = np.log1p(training_data.values).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(raw, bins=100)
    axes[0].set_yscale("log")
    axes[0].set_title("Raw Distribution (training)")
    axes[0].set_xlabel("Value of gene")
    axes[0].set_ylabel("Frequency (log scale)")

    axes[1].hist(log_raw, bins=100)
    axes[1].set_yscale("log")
    axes[1].set_title("Log1p Distribution (training)")
    axes[1].set_xlabel("log1p(Value of gene)")
    axes[1].set_ylabel("Frequency (log scale)")

    plt.tight_layout()
    plt.show()


def prepare_features(training_data, test_data, var_quantile=0.5):
    """
    Log1p then keep high-variance genes then standardize.

    Returns:
      Xtr_z, Xte_z, mask, scaler
    """
    Xtr_log = np.log1p(training_data.values)
    Xte_log = np.log1p(test_data.values)

    gene_var = Xtr_log.var(axis=0)
    threshold = np.quantile(gene_var, var_quantile)
    mask = gene_var >= threshold

    Xtr_filt = Xtr_log[:, mask]
    Xte_filt = Xte_log[:, mask]

    print("Number of genes before filtering:", training_data.shape[1])
    print("Number of genes after filtering :", Xtr_filt.shape[1])

    scaler = StandardScaler().fit(Xtr_filt)
    Xtr_z = scaler.transform(Xtr_filt)
    Xte_z = scaler.transform(Xte_filt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(Xtr_filt.flatten(), bins=100)
    axes[0].set_yscale("log")
    axes[0].set_title("Before Standardization")
    axes[1].hist(Xtr_z.flatten(), bins=100)
    axes[1].set_yscale("log")
    axes[1].set_title("After Standardization")
    plt.tight_layout()
    plt.show()

    return Xtr_z, Xte_z, mask, scaler
