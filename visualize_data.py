import matplotlib.pyplot as plt
import numpy as np


def show_data_distribution(ra_tr, sle_tr, healthy_tr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, df, title in zip(
        axes,
        [healthy_tr, ra_tr, sle_tr],
        ["Healthy (train)", "RA (train)", "SLE (train)"],
    ):
        values = df.values.flatten()
        ax.hist(values, bins=100)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Expression")
        ax.set_ylabel("Frequency (log)")

    plt.tight_layout()
    plt.show()
