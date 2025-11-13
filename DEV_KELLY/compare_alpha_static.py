# experiments/run_static_compare_alpha.py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import math
from dataclasses import dataclass

# ✅ permettre l'import de src/*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models import Player, ResourceOwner, synchronous_best_response  # on n'utilise pas Simulator ici

EPS_UTIL = 1e-12  # pour éviter log(0)

@dataclass
class RunData:
    times: np.ndarray
    bids_ts: dict         # pid -> list[float]
    alloc_ts: dict        # pid -> list[float]
    globals: dict         # 'sum_bids','price','welfare','n_players'


def u_alpha(alpha: int, x: float) -> float:
    """Utilité α-fair standard: x^(1-α)/(1-α) (α!=1), log(x+eps) (α=1)."""
    x = max(0.0, x)
    if alpha == 1:
        return math.log(x + EPS_UTIL)
    return (x ** (1.0 - alpha)) / (1.0 - alpha) if x > 0 else 0.0


def build_players(n: int, lam: float, alpha: int, a_values: np.ndarray, rng: np.random.Generator) -> dict:
    """Construit un dict id->Player avec mêmes 'a' pour tous les α (comparaison équitable)."""
    players = {}
    for pid in range(1, n + 1):
        a_i = float(a_values[pid - 1])
        # bids init petits mais positifs (évite log(0) via allocations nulles)
        bid0 = float(rng.uniform(0.5, 1.5))
        players[pid] = Player(
            id=pid, a=a_i, budget=4000.0, eps=1e-3, lam=lam,
            alpha=alpha, bid=bid0, alive=True
        )
    return players


def run_static(n: int, alpha: int, tmax: float, dt: float, lam: float, C: float,
               delta: float, seed: int, a_values: np.ndarray) -> RunData:
    """
    Simulation statique: N joueurs fixes, best-response synchronisé à intervalle constant.
    """
    rng = np.random.default_rng(seed)
    owner = ResourceOwner(C=C, delta=delta)
    players = build_players(n, lam, alpha, a_values, rng)

    steps = int(round(tmax / dt))
    times = np.linspace(0.0, tmax, steps + 1)

    bids_ts = {pid: [] for pid in range(1, n + 1)}
    alloc_ts = {pid: [] for pid in range(1, n + 1)}

    g_sum_bids, g_price, g_welfare, g_n = [], [], [], []

    for k in range(steps + 1):
        # 1) meilleure réponse synchronisée
        synchronous_best_response(players, owner)

        # 2) allocations et enregistrements
        bids_alive = {i: p.bid for i, p in players.items() if p.alive}
        alloc = owner.allocate(bids_alive) if bids_alive else {}

        for pid in range(1, n + 1):
            b = players[pid].bid if players[pid].alive else 0.0
            x = alloc.get(pid, 0.0)
            bids_ts[pid].append(b)
            alloc_ts[pid].append(x)

        sum_b = float(sum(bids_alive.values())) if bids_alive else 0.0
        price = (sum_b / C) if C > 0 else 0.0
        n_players = len(bids_alive)

        # welfare instantané
        welfare = 0.0
        if bids_alive:
            for i in bids_alive.keys():
                ai = players[i].a
                xi = alloc.get(i, 0.0)
                welfare += ai * u_alpha(alpha, xi)
            welfare -= lam * sum_b

        g_sum_bids.append(sum_b)
        g_price.append(price)
        g_welfare.append(welfare)
        g_n.append(n_players)

    globals_dict = {
        "sum_bids": np.asarray(g_sum_bids, dtype=float),
        "price": np.asarray(g_price, dtype=float),
        "welfare": np.asarray(g_welfare, dtype=float),
        "n_players": np.asarray(g_n, dtype=int),
    }
    return RunData(times=np.asarray(times), bids_ts=bids_ts, alloc_ts=alloc_ts, globals=globals_dict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Nombre de joueurs (fixes)")
    ap.add_argument("--tmax", type=float, default=200.0, help="Horizon (secondes simulées)")
    ap.add_argument("--dt_update", type=float, default=0.5, help="Pas de mise à jour BR synchronisé")
    ap.add_argument("--lam", type=float, default=1.0, help="Coût des bids (λ)")
    ap.add_argument("--C", type=float, default=10.0, help="Capacité totale de ressource")
    ap.add_argument("--delta", type=float, default=0.0, help="Décote éventuelle (si utilisée dans ResourceOwner)")
    ap.add_argument("--seed", type=int, default=123, help="Graine RNG")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parent

    # mêmes 'a_i' pour tous les α → comparaison propre
    rng_a = np.random.default_rng(args.seed)
    a_values = rng_a.uniform(60.0, 140.0, size=args.n)   # valeurs/poids des joueurs

    results = {}
    summary_path = outdir / "simulation_summary_static.txt"

    with open(summary_path, "w", encoding="utf-8") as log:
        log.write("=========== SIMULATION STATIQUE — comparaison α ∈ {0,1,2} ===========\n")
        log.write(f"N={args.n}, C={args.C}, λ={args.lam}, tmax={args.tmax}, dt={args.dt_update}\n\n")

        for alpha in [0, 1, 2]:
            run = run_static(
                n=args.n, alpha=alpha, tmax=args.tmax, dt=args.dt_update,
                lam=args.lam, C=args.C, delta=args.delta, seed=args.seed + alpha, a_values=a_values
            )
            results[alpha] = run

            # Bilans globaux
            sum_bids_time = float(np.sum(run.globals["sum_bids"])) * args.dt_update  # ∫ Σb dt
            welfare_mean = float(np.mean(run.globals["welfare"]))
            price_mean = float(np.mean(run.globals["price"]))

            log.write(f"--- GLOBAL α={alpha} ---\n")
            log.write(f"• Ressource totale disponible C : {args.C:.2f} u\n")
            log.write(f"• Argent total dépensé (Σ b·Δt) : {sum_bids_time:.2f} crédits\n")
            log.write(f"• Prix moyen p̄(t) = Σb/C : {price_mean:.4f} crédits/u\n")
            log.write(f"• Welfare moyen W̄(t) : {welfare_mean:.4f} unités d’utilité\n\n")

            # Bilans par joueur (money_spent, alloc_cumulee, rendement), tri par rendement
            stats = []
            for pid in range(1, args.n + 1):
                money_i = args.dt_update * float(np.sum(run.bids_ts[pid]))
                alloc_cum_i = args.dt_update * float(np.sum(run.alloc_ts[pid]))
                rendement = (alloc_cum_i / money_i) if money_i > 0 else 0.0
                stats.append((pid, money_i, alloc_cum_i, rendement))

            stats.sort(key=lambda x: x[3], reverse=True)

            log.write(f"   id |   money_spent (cr) | alloc_cumulee (u·s) | rendement (u·s/cr)\n")
            log.write(f"---------------------------------------------------------------------\n")
            for pid, mny, xcum, r in stats:
                log.write(f"  {pid:3d} | {mny:17.2f} | {xcum:17.4f} | {r:14.6f}\n")
            log.write("\n")

    # ========= GRAPHIQUES PAR JOUEUR (bids & allocations, overlay α) =========
    for pid in range(1, args.n + 1):
        plt.figure(figsize=(8, 5))
        for alpha, run in results.items():
            plt.plot(run.times, run.bids_ts[pid], label=f"bid α={alpha}")
            plt.plot(run.times, run.alloc_ts[pid], linestyle="--", label=f"alloc α={alpha}")
        plt.title(f"Joueur {pid} — Bids & Allocations (statique, comparaison α)")
        plt.xlabel("Temps simulé (s)")
        plt.ylabel("Valeur (crédits / unités)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(outdir / f"static_player{pid}_compare_alpha.png")
        plt.close()

    # ========= GRAPHIQUES GLOBAUX (overlay α) =========
    def plot_global(attr, title, ylabel, fname):
        plt.figure(figsize=(9, 5))
        for alpha, run in results.items():
            plt.plot(run.times, run.globals[attr], label=f"α={alpha}")
        plt.title(title)
        plt.xlabel("Temps simulé (s)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(outdir / fname)
        plt.close()

    plot_global("sum_bids", "Σ des enchères — statique (comparaison α)", "Σ des enchères (crédits)", "static_global_sum_bids.png")
    plot_global("price", "Prix p(t)=Σ b / C — statique (comparaison α)", "Prix p(t) (crédits/u)", "static_global_price.png")
    plot_global("n_players", "Nombre de joueurs actifs — statique", "N (joueurs)", "static_global_n_players.png")
    plot_global("welfare", "Welfare total W(t) — statique (comparaison α)", "W(t) (unités d’utilité)", "static_global_welfare.png")

    print(f"✅ Fini. Résumé écrit dans: {outdir / 'simulation_summary_static.txt'}")


if __name__ == "__main__":
    main()