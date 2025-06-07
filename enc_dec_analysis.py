import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from scipy.special import digamma, gammaln
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Paths
base_dir   = "/work/data/ssd_results"
output_dir = os.path.join("work/project", "tsne_entropy_kl_plots")

# Clean output directory
for fname in os.listdir(output_dir):
    if fname.endswith("_plots.pdf") or fname.endswith("_plots.png") or fname.endswith("_evolution.mp4"):
        try:
            os.remove(os.path.join(output_dir, fname))
        except OSError as e:
            print(f"Errore rimuovendo {fname}: {e}")
os.makedirs(output_dir, exist_ok=True)

# Configurazione varianti
opts   = ["meta", "pyra", "moh"]
wheres = ["full", "enc", "dec"]

# Mappatura colori per fase
phase_colors = {
    "full": "#b0d08f",  # scurito rispetto a #e6f5d0
    "enc":  "#7bb34e",  # scurito rispetto a #a1d76a
    "dec":  "#33681c"   # scurito rispetto a #4d9221
}

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
    return [f"{net}_{opt}_{where}_{size}_{ds}"
            for opt in opts for where in wheres]

def split_label(var):
    _, opt, where, *_ = var.split("_")
    return opt.capitalize(), where.lower()

# Raggruppa tutti gli emb.pkl
if len([f for f in os.listdir(output_dir) if f.endswith("_plots.png")]) < 12:
    groups = {}
    for root, _, files in os.walk(base_dir):
        if "emb.pkl" in files:
            net, sz, ds = parse_folder(os.path.basename(root))
            groups.setdefault((net, sz, ds), []).append(os.path.join(root, "emb.pkl"))

    for (net, size, ds), paths in tqdm(groups.items(), desc="Processing groups"):
        baseline_name = f"{net}_{size}_{ds}"

        ent_labels, ent_vals = [], []
        kl_labels,  kl_vals  = [], []

        # Carica embeddings e calcola entropie, salta la baseline
        for pkl_path in sorted(paths):
            var = os.path.basename(os.path.dirname(pkl_path))
            if var == baseline_name:
                continue

            with open(pkl_path, "rb") as f:
                Xv = pickle.load(f)["embeddings"]

            # Gaussian differential entropy
            cov = np.cov(Xv, rowvar=False)
            _, logdet = np.linalg.slogdet(cov)
            d = Xv.shape[1]
            H_gauss = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            ent_labels.append(var)
            ent_vals.append(H_gauss)

            # Kozachenko–Leonenko entropy
            H_kl = estimate_kl_entropy(Xv)
            kl_labels.append(var)
            kl_vals.append(H_kl)

        # Prepara dati ordinati
        order       = get_variant_order(net, size, ds)
        ent_dict    = dict(zip(ent_labels, ent_vals))
        kl_dict     = dict(zip(kl_labels, kl_vals))
        ent_ord     = [(v, ent_dict[v]) for v in order if v in ent_dict]
        kl_ord      = [(v, kl_dict[v]) for v in order if v in kl_dict]
        ent_vars, ent_values = zip(*ent_ord)
        kl_vars,  kl_values  = zip(*kl_ord)

        # Split labels e colori
        ent_split = [split_label(v) for v in ent_vars]  # (Opt, phase)
        kl_split  = [split_label(v) for v in kl_vars]
        ent_colors = [phase_colors[phase] for _, phase in ent_split]
        kl_colors  = [phase_colors[phase] for _, phase in kl_split]

        # Disegna figure
        fig = plt.figure(figsize=(20, 10))
        gs  = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.6], wspace=0.15)
        ax_ent = fig.add_subplot(gs[0, 0])
        ax_kl  = fig.add_subplot(gs[0, 1])

        # --- Gaussian entropy ---
        N_ent = len(ent_vars)
        y_ent = np.arange(N_ent)[::-1]
        ent_abs = np.abs(ent_values)
        ax_ent.barh(y_ent, ent_abs, height=0.6,
                    color=ent_colors, alpha=0.9)
        ax_ent.set_yticks([]); ax_ent.invert_yaxis()
        ax_ent.set_xlim(0, max(ent_abs)*1.1)
        ax_ent.set_xlabel("|H| (nats)", fontsize=14)
        ax_ent.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Etichette multilivello
        opt_groups = {}
        for i, (opt, phase) in enumerate(ent_split):
            y = y_ent[i]
            ax_ent.text(-0.03, y, phase.capitalize(), rotation=90,
                        transform=ax_ent.get_yaxis_transform(),
                        va='center', ha='center', fontsize=12)
            opt_groups.setdefault(opt, []).append(y)
        for opt, ys in opt_groups.items():
            ax_ent.text(-0.07, np.mean(ys), opt, rotation=90,
                        transform=ax_ent.get_yaxis_transform(),
                        va='center', ha='center',
                        fontsize=12, fontweight='bold')

        # Separatori fra i gruppi di ottimizzazione
        for i in range(1, len(opts)):
            sep = N_ent - i*len(wheres) - 0.5
            ax_ent.axhline(y=sep, color='gray', linestyle='--',
                           linewidth=1.2, alpha=0.8)

        # Valori sulle barre
        for i, val in enumerate(ent_abs):
            ax_ent.text(val*0.5, y_ent[i], f"{ent_values[i]:.2f}",
                        va='center', ha='center',
                        fontsize=12, color='white', fontweight='bold')


        # --- KL entropy ---
        N_kl = len(kl_vars)
        y_kl = np.arange(N_kl)[::-1]
        kl_abs = np.abs(kl_values)
        ax_kl.barh(y_kl, kl_abs, height=0.6,
                   color=kl_colors, alpha=0.9)
        ax_kl.set_yticks([]); ax_kl.invert_yaxis()
        ax_kl.set_xlim(0, max(kl_abs)*1.1)
        ax_kl.set_xlabel("|H_KL| (nats)", fontsize=14)
        ax_kl.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        opt_groups_kl = {}
        for i, (opt, phase) in enumerate(kl_split):
            y = y_kl[i]
            ax_kl.text(-0.03, y, phase.capitalize(), rotation=90,
                       transform=ax_kl.get_yaxis_transform(),
                       va='center', ha='center', fontsize=12)
            opt_groups_kl.setdefault(opt, []).append(y)
        for opt, ys in opt_groups_kl.items():
            ax_kl.text(-0.07, np.mean(ys), opt, rotation=90,
                       transform=ax_kl.get_yaxis_transform(),
                       va='center', ha='center',
                       fontsize=12, fontweight='bold')

        # Separatori fra i gruppi di ottimizzazione (KL)
        for i in range(1, len(opts)):
            sep = N_kl - i*len(wheres) - 0.5
            ax_kl.axhline(y=sep, color='gray', linestyle='--',
                          linewidth=1.2, alpha=0.8)

        # Valori sulle barre KL
        for i, val in enumerate(kl_abs):
            ax_kl.text(val*0.5, y_kl[i], f"{kl_values[i]:.2f}",
                       va='center', ha='center',
                       fontsize=12, color='white', fontweight='bold')

        # Salvataggio
        plt.subplots_adjust(bottom=0.2, right=0.85)
        fig.savefig(os.path.join(
            output_dir, f"e_{net}_{size}_{ds}_plots.pdf"),
            dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

print("✔ All plots saved to", output_dir)
