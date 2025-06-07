import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.special import digamma, gammaln
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# podman run -v /home/schiavella/thesis_claudio_lorenzo/:/work/project -v /mnt/ssd1/schiavella/:/work/data --device nvidia.com/gpu=all --ipc host docker.io/claudioschi21/thesis_alcor_cuda11.8:latest /usr/bin/python3 work/project/enc_dec_analysis.py

# Parameters
base_dir = "/work/data/ssd_results"
output_dir = os.path.join("work/project", "tsne_entropy_kl_plots")
for fname in os.listdir(output_dir):
    if fname.endswith("_plots.pdf") or fname.endswith("_plots.png") or fname.endswith("_evolution.mp4"):
        try:
            os.remove(os.path.join(output_dir, fname))
        except OSError as e:
            print(f"Errore rimuovendo {fname}: {e}")
os.makedirs(output_dir, exist_ok=True)


def parse_folder(name):
    parts = name.split("_")
    return parts[0], parts[-2], parts[-1]


def estimate_kl_entropy(X, k=5, eps=1e-10):
    n, d = X.shape
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    r_k = np.maximum(distances[:, -1], eps)
    log_vol = (d/2) * math.log(math.pi) - gammaln(d/2 + 1)
    return digamma(n) - digamma(k) + log_vol + (d/n) * np.sum(np.log(r_k))


def get_variant_order(net, size, ds):
    baseline = f"{net}_{size}_{ds}"
    opts = ["meta", "pyra", "moh"]
    wheres = ["full", "enc", "dec"]
    return [baseline] + [f"{net}_{opt}_{where}_{size}_{ds}"
                         for opt in opts for where in wheres]

# Styles
markers = ['o','s','^','v','<','>','*','x','+','d']
colors = plt.get_cmap('tab10').colors

# Loop per gruppi e creazione dei plot
if len([f for f in os.listdir(output_dir) if f.endswith("_plots.png")]) < 12:
    groups = {}
    for root, _, files in os.walk(base_dir):
        if "emb.pkl" in files:
            net, sz, ds = parse_folder(os.path.basename(root))
            groups.setdefault((net, sz, ds), []).append(os.path.join(root, "emb.pkl"))

    for (net, size, ds), paths in tqdm(groups.items(), desc="Processing groups"):

        # Raccolta embeddings ed entropie
        X_parts, labels = [], []
        ent_labels, ent_vals = [], []
        kl_labels, kl_vals = [], []

        for pkl_path in sorted(paths):
            var = os.path.basename(os.path.dirname(pkl_path))
            with open(pkl_path, "rb") as f:
                Xv = pickle.load(f)["embeddings"]
            X_parts.append(Xv)
            labels += [var] * len(Xv)

            # Gaussian differential entropy
            cov = np.cov(Xv, rowvar=False)
            _, logdet = np.linalg.slogdet(cov)
            d = Xv.shape[1]
            H_gauss = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            ent_labels.append(var); ent_vals.append(H_gauss)

            # Kozachenko–Leonenko entropy
            H_kl = estimate_kl_entropy(Xv)
            kl_labels.append(var); kl_vals.append(H_kl)

        # Unico array per i bar plot
        order = get_variant_order(net, size, ds)
        ent_dict = dict(zip(ent_labels, ent_vals))
        kl_dict = dict(zip(kl_labels, kl_vals))
        ent_ord = [(v, ent_dict[v]) for v in order if v in ent_dict]
        kl_ord = [(v, kl_dict[v]) for v in order if v in kl_dict]
        ent_vars, ent_values = zip(*ent_ord)
        kl_vars, kl_values = zip(*kl_ord)

        # Funzione per etichette a due righe
        def split_label(var):
            parts = var.split("_")
            if len(parts) == 5:
                return parts[1].capitalize(), parts[2].capitalize()
            else:
                return "Baseline", ""

        # Prepara le etichette per i due livelli
        ent_labels_split = [split_label(v) for v in ent_vars]
        kl_labels_split = [split_label(v) for v in kl_vars]

        # Crea figura 1x2
        fig = plt.figure(figsize=(20, 10))
        # fig.suptitle(f"{net.upper()} | {size} | {ds}", fontsize=24, fontweight='bold')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.6], wspace=0.15)
        ax_ent = fig.add_subplot(gs[0, 0])
        ax_kl = fig.add_subplot(gs[0, 1])

        # Gaussian entropy barh
        y_ent = np.arange(len(ent_vars))[::-1]
        ent_abs = np.abs(ent_values)
        ent_colors = [colors[i % len(colors)] for i in range(len(ent_vars))]
        ax_ent.barh(y_ent, ent_abs, height=0.6, color=ent_colors, alpha=0.9)
        ax_ent.set_yticks(y_ent)
        
        # Crea etichette multi-livello per entropy
        ax_ent.set_yticklabels([])  # Rimuovi le etichette standard
        ax_ent.invert_yaxis()
        ax_ent.set_xlim(0, max(ent_abs) * 1.1)
        
        # Aggiungi etichette personalizzate per entropy
        opt_groups = {}
        for i, (opt, where) in enumerate(ent_labels_split):
            y_pos = y_ent[i]
            if opt != "Baseline":
                # Seconda colonna (posizione) - sempre mostrata
                ax_ent.text(-0.03, y_pos, where, rotation=90, transform=ax_ent.get_yaxis_transform(),
                           va='center', ha='center', fontsize=12)
                # Raggruppa per ottimizzazione
                if opt not in opt_groups:
                    opt_groups[opt] = []
                opt_groups[opt].append(y_pos)
            else:
                # Baseline orizzontale
                ax_ent.text(-0.03, y_pos, opt, rotation=90, transform=ax_ent.get_yaxis_transform(),
                           va='center', ha='center', fontsize=11)
        
        # Prima colonna (ottimizzazione) - solo al centro di ogni gruppo
        for opt, y_positions in opt_groups.items():
            center_y = np.mean(y_positions)
            ax_ent.text(-0.07, center_y, opt, rotation=90, transform=ax_ent.get_yaxis_transform(),
                       va='center', ha='center', fontsize=12, fontweight='bold')

        ax_ent.set_xlabel("|H| (nats)", fontsize=14)
        # ax_ent.set_title("Gaussian Differential Entropy", fontsize=16)
        ax_ent.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        for i, val in enumerate(ent_abs):
            ax_ent.text(val * 0.5, y_ent[i], f"{ent_values[i]:.2f}",
                        va='center', ha='center', fontsize=12,
                        color='white', fontweight='bold')

        # Linee di separazione
        if len(ent_vars) > 1:
            baseline_pos = len(y_ent) - 1
            if baseline_pos > 0:
                ax_ent.axhline(y=baseline_pos - 0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
                for i in range(1, 3):
                    sep = baseline_pos - i * 3 - 0.5
                    if sep >= 0:
                        ax_ent.axhline(y=sep, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)

        # KL entropy barh
        y_kl = np.arange(len(kl_vars))[::-1]
        kl_abs = np.abs(kl_values)
        kl_colors = [colors[i % len(colors)] for i in range(len(kl_vars))]
        ax_kl.barh(y_kl, kl_abs, height=0.6, color=kl_colors, alpha=0.9)
        ax_kl.set_yticks(y_kl)
        
        # Crea etichette multi-livello per KL
        ax_kl.set_yticklabels([])  # Rimuovi le etichette standard
        ax_kl.invert_yaxis()
        ax_kl.set_xlim(0, max(kl_abs) * 1.1)
        
        # Aggiungi etichette personalizzate per KL
        opt_groups_kl = {}
        for i, (opt, where) in enumerate(kl_labels_split):
            y_pos = y_kl[i]
            if opt != "Baseline":
                # Seconda colonna (posizione) - sempre mostrata
                ax_kl.text(-0.03, y_pos, where, rotation=90, transform=ax_kl.get_yaxis_transform(),
                          va='center', ha='center', fontsize=12)
                # Raggruppa per ottimizzazione
                if opt not in opt_groups_kl:
                    opt_groups_kl[opt] = []
                opt_groups_kl[opt].append(y_pos)
            else:
                # Baseline orizzontale
                ax_kl.text(-0.03, y_pos, opt, rotation=90, transform=ax_kl.get_yaxis_transform(),
                          va='center', ha='center', fontsize=11)
        
        # Prima colonna (ottimizzazione) - solo al centro di ogni gruppo
        for opt, y_positions in opt_groups_kl.items():
            center_y = np.mean(y_positions)
            ax_kl.text(-0.07, center_y, opt, rotation=90, transform=ax_kl.get_yaxis_transform(),
                      va='center', ha='center', fontsize=12, fontweight='bold')

        # Linee di separazione per KL
        if len(kl_vars) > 1:
            baseline_pos = len(y_kl) - 1
            if baseline_pos > 0:
                ax_kl.axhline(y=baseline_pos - 0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
                for i in range(1, 3):
                    sep = baseline_pos - i * 3 - 0.5
                    if sep >= 0:
                        ax_kl.axhline(y=sep, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
        
        ax_kl.set_xlabel("|H_KL| (nats)", fontsize=14)
        # ax_kl.set_title("Non-parametric K-L Entropy", fontsize=16)
        ax_kl.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        for i, val in enumerate(kl_abs):
            ax_kl.text(val * 0.5, y_kl[i], f"{kl_values[i]:.2f}",
                       va='center', ha='center', fontsize=12,
                       color='white', fontweight='bold')

        # Salva e chiudi
        plt.subplots_adjust(bottom=0.2, right=0.85)
        fig.savefig(os.path.join(output_dir, f"e_{net}_{size}_{ds}_plots.pdf"), dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

print("✔ All plots saved to", output_dir)