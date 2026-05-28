""" Make ternary graphs. 
Most plots will be hexbin heatmaps due to large number of points.
"""

import math
import matplotlib.pyplot as plt
import mpltern
import numpy as np


def _make_ternary_subplots(n: int):
    """ Create a matplotlib plot with n ternary subplots """
    num_cols = math.ceil(math.sqrt(n))
    num_rows = math.ceil(n / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12,10),
                             subplot_kw=dict(projection="ternary"))
    axes = np.array(axes).reshape(-1)
    return fig, axes


def plot_overall_distribution(connectome, nsamples, file):
    """ Save a ternary plot of the overall neurotransmitter prob distr """
    gaba, ach, other = connectome[["gaba", "ach", "other"]].compute().to_numpy().T
    #print(probs)
    #gaba, ach, other = probs[0], probs[1], probs[2]
    ax = plt.subplot(projection="ternary")
    ax.hexbin(gaba, ach, other)
    ax.set_tlabel("GABA Probability")
    ax.set_llabel("Acetylcholine Probability")
    ax.set_rlabel("Other Neurotransmitter Probability")
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    ax.set_title("Overall Nodule Neurotransmitter Probability "
                 f"(Sample Size = {nsamples})", pad=50, size=14)
    plt.tight_layout()
    plt.savefig(file, format="svg", dpi=300)
    

def plot_mean_nt_probs_by_neuropil_size(connectome, file):
    """ Create one ternary scatter plot where point colour corresponds to number 
    of synapses in a neuropil, and each point is the average probability within a
    neuropil. connectome dataframe must be pre-aggregated. """
    neuropil_size, gaba, ach, other = connectome[
        ["neuropil_size", "gaba_mean", "ach_mean", "other_mean"]
        ].compute().to_numpy().T
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(projection="ternary")
    scattered = ax.scatter(gaba, ach, other, c=neuropil_size)
    ax.set_tlabel("GABA Probability")
    ax.set_llabel("Acetylcholine Probability")
    ax.set_rlabel("Other Neurotransmitter Probability")
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    ax.set_title("Mean Neurotransmitter Probabilities by Neuropil Size", size=14, pad=50)
    colour_bar = fig.colorbar(scattered, shrink=0.7)
    colour_bar.ax.set_title("Synapse Count", size=10)
    plt.tight_layout()
    plt.savefig(file, format="svg", dpi=300)


def plot_variance_by_neuropil_size(connectome, file):
    """ Create a bar chart showing neuropil neurotransmitter variances, sorted 
    by neuropil size """
    sorted_connectome = connectome.sort_values("neuropil_size").reset_index(drop=False)
    neuropil, gaba, ach, other = sorted_connectome[
        ["neuropil", "gaba_var", "ach_var", "other_var"]].compute().to_numpy().T
    
    # Set bar positions
    bar_width = 0.25
    r = np.arange(len(gaba))
    r2 = r + bar_width
    r3 = r2 + bar_width
    
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    ax.bar(r, gaba, color="blue", width=bar_width, label="GABA")
    ax.bar(r2, ach, color="yellow", width=bar_width, label="ACH")
    ax.bar(r3, other, color="pink", width=bar_width, label="OTHER")
    ax.set_xlabel("Neuropil")
    ax.set_xticks(r2)
    ax.set_xticklabels(neuropil)
    ax.legend(title="Neurotransmitter Probability Variances", ncol=3)
    plt.tight_layout()
    plt.savefig(file, format="svg", dpi=300)


def plot_hex_per_neuropil(connectome, file: str):
    """ Create ternary hexbin plots showing the overall distributions of actual
    probabilities for all synapses in the dask dataframes in connectomes list """
    neuropils = list(connectome["neuropil"].unique().compute())
    grouped = connectome.groupby("neuropil")
    num_plots = len(neuropils)
    
    fig, axes = _make_ternary_subplots(num_plots)
    
    for i, ax in enumerate(axes[:num_plots]):
        
        # Compute neuropil i's probabilities into a numpy array
        group = grouped.get_group(neuropils[i])
        gaba, ach, other = group[["gaba", "ach", "other"]].compute().to_numpy().T
        
        # Plot hexbin
        hb = ax.hexbin(gaba, ach, other, gridsize=8)
        ax.set_tlabel("GABA")
        ax.set_llabel("ACH")
        ax.set_rlabel("OTHER")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        ax.set_title(neuropils[i])
    
    # Hide unused axes
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")
    plt.tight_layout() # fit subplots to figure size
    plt.savefig(file, format="svg", dpi=300)