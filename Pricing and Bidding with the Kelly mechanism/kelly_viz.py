
"""
kelly_viz.py
------------
Visualisation de l'historique du simulateur du mécanisme de Kelly.
Utilisation:
    python kelly_viz.py --csv kelly_history.csv

Le script génère trois figures (une par métrique) et peut aussi les
sauvegarder au format PNG via --outdir.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def make_plots(df, outdir=None, show=True):
    # Figure 1: number of players over time
    plt.figure()
    plt.plot(df["time"], df["n_players"])
    plt.xlabel("time")
    plt.ylabel("n_players")
    plt.title("Number of players over time")
    plt.tight_layout()
    if outdir:
        plt.savefig(Path(outdir) / "n_players.png", dpi=150)

    # Figure 2: total bids over time
    plt.figure()
    plt.plot(df["time"], df["total_bids"])
    plt.xlabel("time")
    plt.ylabel("total_bids")
    plt.title("Total bids over time")
    plt.tight_layout()
    if outdir:
        plt.savefig(Path(outdir) / "total_bids.png", dpi=150)

    # Figure 3: social welfare over time
    plt.figure()
    plt.plot(df["time"], df["welfare"])
    plt.xlabel("time")
    plt.ylabel("welfare")
    plt.title("Social welfare over time")
    plt.tight_layout()
    if outdir:
        plt.savefig(Path(outdir) / "welfare.png", dpi=150)

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="kelly_history.csv",
                        help="Chemin du CSV produit par le simulateur")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Dossier de sortie pour enregistrer les PNG (optionnel)")
    parser.add_argument("--no-show", action="store_true",
                        help="Ne pas afficher les figures (utile en batch)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    make_plots(df, outdir=args.outdir, show=not args.no_show)

if __name__ == "__main__":
    main()
