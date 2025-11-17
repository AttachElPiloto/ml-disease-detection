import pandas as pd


def load_tsv(path: str) -> pd.DataFrame:
    """Read a TSV file with genes as columns and samples as rows."""
    return pd.read_csv(path, sep="\t", index_col=0)


def get_training_and_test_data():
    """Load RA / SLE / Healthy train & test sets and align them on common genes."""

    ra_tr = load_tsv("./training_data/ra_train_data.tsv")
    ra_te = load_tsv("./test_data/ra_test_data.tsv")

    sle_tr = load_tsv("./training_data/sle_train_data.tsv")
    sle_te = load_tsv("./test_data/sle_test_data.tsv")

    healthy_tr = load_tsv("./training_data/healthy_train_data.tsv")
    healthy_te = load_tsv("./test_data/healthy_test_data.tsv")

    # intersect genes across all datasets
    genes = (
        set(ra_tr.columns)
        & set(ra_te.columns)
        & set(sle_tr.columns)
        & set(sle_te.columns)
        & set(healthy_tr.columns)
        & set(healthy_te.columns)
    )
    genes = sorted(genes)

    print("Number of genes per dataset:")
    print("RA train :", ra_tr.shape[1])
    print("RA test  :", ra_te.shape[1])
    print("SLE train:", sle_tr.shape[1])
    print("SLE test :", sle_te.shape[1])
    print("Healthy train :", healthy_tr.shape[1])
    print("Healthy test  :", healthy_te.shape[1])
    print("\nGenes in intersection:", len(genes))

    training_data = pd.concat(
        [
            healthy_tr.loc[:, genes].assign(label="Healthy"),
            ra_tr.loc[:, genes].assign(label="RA"),
            sle_tr.loc[:, genes].assign(label="SLE"),
        ]
    )

    test_data = pd.concat(
        [
            healthy_te.loc[:, genes].assign(label="Healthy"),
            ra_te.loc[:, genes].assign(label="RA"),
            sle_te.loc[:, genes].assign(label="SLE"),
        ]
    )

    y_tr = training_data.pop("label")
    y_te = test_data.pop("label")

    print("\nTraining data shape:", training_data.shape)
    print("Test data shape:", test_data.shape)

    return healthy_tr, ra_tr, sle_tr, training_data, test_data, y_tr, y_te
