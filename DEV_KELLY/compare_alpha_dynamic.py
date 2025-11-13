import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import math

# ✅ import du dossier "src/" même sans exécuter avec -m
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from simulator import Simulator
from models import Player, ResourceOwner, synchronous_best_response

MAX_PLAYERS = 10  # nombre maximum de joueurs simultanés
EPS_UTIL = 1e-12  # pour éviter log(0)


# -------------------------------------------------------------
# Tirage exponentiel (temps d'arrivée et de départ aléatoire)
# -------------------------------------------------------------
def sample_exponential(rng, rate: float) -> float:
    if rate <= 0:
        return float("inf")
    return rng.expovariate(rate)


# -------------------------------------------------------------
# Fonction d’utilité α-fair
# -------------------------------------------------------------
def u_alpha(alpha: int, x: float) -> float:
    """Utilité α-fair standard: x^(1-α)/(1-α) (α!=1), log(x+eps) (α=1)."""
    x = max(0.0, x)
    if alpha == 1:
        return math.log(x + EPS_UTIL)
    return (x ** (1.0 - alpha)) / (1.0 - alpha) if x > 0 else 0.0


# -------------------------------------------------------------
# Simulation dynamique pour un alpha donné
# -------------------------------------------------------------
def run_dynamic(alpha, A, B, tmax, dt_update, lam, C, delta, seed, log_event):
    sim = Simulator(seed=seed)
    rng = sim.random
    owner = ResourceOwner(C=C, delta=delta)

    players = {}
    alive_ids = set()
    next_uid = 1

    times = []
    bids_ts = {i: [] for i in range(1, MAX_PLAYERS + 1)}
    alloc_ts = {i: [] for i in range(1, MAX_PLAYERS + 1)}

    # séries globales
    g_sum_bids, g_price, g_n_players, g_welfare = [], [], [], []
    g_alloc_total, g_perf = [], []   # nouvelle : allocation totale et performance économique

    # Planifie la prochaine arrivée
    def schedule_next_arrival(now):
        t = now + sample_exponential(rng, A)
        if t < tmax:
            sim.schedule(t, "ARRIVAL")

    # Crée un nouveau joueur
    def new_player(uid_raw):
        mapped_id = ((uid_raw - 1) % MAX_PLAYERS) + 1
        a_i = (100.0 / max(1, mapped_id)) * rng.uniform(0.9, 1.1)
        budget = 4000.0
        eps = 1e-3
        return mapped_id, Player(
            id=mapped_id, a=a_i, budget=budget, eps=eps,
            lam=lam, alpha=alpha, bid=1.0, alive=True
        )

    # ---------------------------------------------------------
    # Événement : arrivée d’un joueur
    # ---------------------------------------------------------
    @sim.on("ARRIVAL")
    def handle_arrival(evt):
        nonlocal next_uid
        if len(alive_ids) < MAX_PLAYERS:
            mapped_id, p = new_player(next_uid)
            next_uid += 1
            players[mapped_id] = p
            alive_ids.add(mapped_id)
            log_event.write(f"[t={sim.time:.2f}] ARRIVAL → joueur {mapped_id} (a={p.a:.2f})\n")

            t_dep = sim.time + sample_exponential(rng, B)
            if t_dep < tmax:
                sim.schedule(t_dep, "DEPARTURE", player_id=mapped_id)

        schedule_next_arrival(sim.time)

    # ---------------------------------------------------------
    # Événement : départ d’un joueur
    # ---------------------------------------------------------
    @sim.on("DEPARTURE")
    def handle_departure(evt):
        pid = evt.payload["player_id"]
        if pid in players:
            players[pid].alive = False
            players[pid].bid = 0.0
            log_event.write(f"[t={sim.time:.2f}] DEPARTURE → joueur {pid}\n")
        alive_ids.discard(pid)

    # ---------------------------------------------------------
    # Événement : mise à jour des enchères (best-response)
    # ---------------------------------------------------------
    @sim.on("BID_UPDATE")
    def handle_bid_update(evt):
        synchronous_best_response(players, owner)
        bids_alive = {i: p.bid for i, p in players.items() if p.alive}
        alloc = owner.allocate(bids_alive) if bids_alive else {}

        # Enregistre l'état courant par joueur
        times.append(sim.time)
        for i in range(1, MAX_PLAYERS + 1):
            b = players[i].bid if (i in players and players[i].alive) else 0.0
            x = alloc.get(i, 0.0)
            bids_ts[i].append(b)
            alloc_ts[i].append(x)

        # Calculs globaux
        sum_b = float(sum(bids_alive.values())) if bids_alive else 0.0
        price = (sum_b / C) if C > 0 else 0.0
        n_players = int(len(bids_alive))

        # allocation totale à cet instant
        alloc_total = float(sum(alloc.values())) if alloc else 0.0

        # performance économique : unités de ressource par crédit dépensé
        perf = alloc_total / sum_b if sum_b > 0 else 0.0

        welfare = 0.0
        if bids_alive:
            for i in bids_alive.keys():
                ai = players[i].a
                xi = alloc.get(i, 0.0)
                welfare += ai * u_alpha(alpha, xi)
            welfare -= lam * sum_b

        g_sum_bids.append(sum_b)
        g_price.append(price)
        g_n_players.append(n_players)
        g_welfare.append(welfare)
        g_alloc_total.append(alloc_total)
        g_perf.append(perf)

        # Écrit dans le log d'événements
        log_event.write(
            f"[t={sim.time:.2f}] BID_UPDATE : N={n_players}, Σb={sum_b:.3f}, "
            f"alloc_tot={alloc_total:.3f}, perf={perf:.4f}, p={price:.3f}, welfare={welfare:.3f}\n"
        )

        # Replanifie la prochaine mise à jour
        t_next = sim.time + dt_update
        if t_next <= tmax:
            sim.schedule(t_next, "BID_UPDATE")

    # Démarrage
    schedule_next_arrival(0.0)
    sim.schedule(0.0, "BID_UPDATE")
    sim.run(until_time=tmax)

    globals_dict = {
        "sum_bids": np.asarray(g_sum_bids, dtype=float),
        "price": np.asarray(g_price, dtype=float),
        "n_players": np.asarray(g_n_players, dtype=int),
        "welfare": np.asarray(g_welfare, dtype=float),
        "alloc_total": np.asarray(g_alloc_total, dtype=float),
        "perf": np.asarray(g_perf, dtype=float),   # performance économique dans le temps
    }
    return times, bids_ts, alloc_ts, globals_dict


# -------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=float, default=0.5)
    ap.add_argument("--B", type=float, default=0.25)
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--dt_update", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parent
    results = {}

    summary_path = outdir / "simulation_summary.txt"
    eventlog_path = outdir / "event_log.txt"

    with open(summary_path, "w", encoding="utf-8") as log, open(eventlog_path, "w", encoding="utf-8") as log_event:
        log.write("====================== LANCEMENT DES SIMULATIONS ======================\n\n")

        for alpha in [0, 1, 2]:
            log.write(f"→ Simulation α={alpha}\n")
            log_event.write(f"\n===== DÉBUT SIMULATION α={alpha} =====\n")
            times, bids_ts, alloc_ts, g = run_dynamic(
                alpha, args.A, args.B, args.tmax, args.dt_update,
                args.lam, args.C, args.delta, args.seed, log_event
            )
            results[alpha] = (times, bids_ts, alloc_ts, g)

            total_resource = args.C
            total_money_spent = float(np.sum(g["sum_bids"])) * args.dt_update
            final_alloc_total = sum(alloc_ts[i][-1] for i in range(1, MAX_PLAYERS + 1) if alloc_ts[i])

            log.write(f"\n--- Résumé GLOBAL α={alpha} ---\n")
            log.write(f"• Ressource totale disponible : {total_resource:.2f} unités\n")
            log.write(f"• Argent total dépensé (Σ b(t)·Δt) : {total_money_spent:.2f} crédits\n")
            log.write(f"• Ressources allouées à la fin : {final_alloc_total:.4f} unités\n")

            # Stats par joueur
            player_stats = []
            for pid in range(1, MAX_PLAYERS + 1):
                money_i = args.dt_update * float(np.sum(bids_ts[pid])) if bids_ts[pid] else 0.0
                alloc_cum_i = args.dt_update * float(np.sum(alloc_ts[pid])) if alloc_ts[pid] else 0.0
                rendement = (alloc_cum_i / money_i) if money_i > 0 else 0.0
                player_stats.append((pid, money_i, alloc_cum_i, rendement))

            player_stats.sort(key=lambda x: x[3], reverse=True)

            log.write(f"\nDétail par joueur (α={alpha}) — trié par rendement décroissant :\n")
            log.write("   id |   money_spent (cr) | alloc_cumulee (u·s) | rendement (u·s/cr)\n")
            log.write("----------------------------------------------------------------------\n")
            for pid, money_i, alloc_cum_i, rendement in player_stats:
                log.write(f"  {pid:3d} | {money_i:17.2f} | {alloc_cum_i:17.4f} | {rendement:14.6f}\n")

            log.write("\n----------------------------------------------------------------\n\n")

    # Graphiques par joueur
    for i in range(1, MAX_PLAYERS + 1):
        plt.figure(figsize=(8, 5))
        for alpha, (times, bids_ts, alloc_ts, g) in results.items():
            plt.plot(times, bids_ts[i], label=f"Bid α={alpha} (crédits)")
            plt.plot(times, alloc_ts[i], linestyle="--", label=f"Alloc α={alpha} (unités)")
        plt.title(f"Joueur {i} — Bids & Allocations (comparaison α)")
        plt.xlabel("Temps simulé (secondes)")
        plt.ylabel("Valeur (crédits / unités)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(outdir / f"player{i}_compare_alpha.png")
        plt.close()

        # Graphiques globaux
    def plot_global(attr, title, ylabel, fname):
        plt.figure(figsize=(9, 5))
        for alpha, (times, _, __, g) in results.items():
            plt.plot(times, g[attr], label=f"α={alpha}")
        plt.title(title)
        plt.xlabel("Temps simulé (secondes)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(outdir / fname)
        plt.close()

    # Comparaisons globales "classiques"
    plot_global(
        "sum_bids",
        "Σ des enchères (t) — comparaison α",
        "Σ des enchères (crédits)",
        "global_compare_sum_bids.png",
    )
    plot_global(
        "price",
        "Prix p(t) = Σ b / C — comparaison α",
        "Prix p(t) (crédits/unité)",
        "global_compare_price.png",
    )
    plot_global(
        "welfare",
        "Welfare total W(t) — comparaison α",
        "W(t) (unités d’utilité)",
        "global_compare_welfare.png",
    )

    # -------- Performance économique CUMULÉE dans le temps --------
    plt.figure(figsize=(9, 5))
    for alpha, (times, _, __, g) in results.items():
        times_arr = np.asarray(times, dtype=float)
        dt = args.dt_update

        # intégrales discrètes ≈ somme * dt
        cum_alloc = np.cumsum(g["alloc_total"]) * dt      # u·s
        cum_bids  = np.cumsum(g["sum_bids"]) * dt         # cr·s

        # pour éviter la division par ~0
        eps = 1e-6
        perf_cum = cum_alloc / (cum_bids + eps)

        plt.plot(times_arr, perf_cum, label=f"α={alpha}")

    plt.title("Performance économique CUMULÉE — comparaison α")
    plt.xlabel("Temps simulé (secondes)")
    plt.ylabel("Perf_cum(t) = (∫ Σ alloc) / (∫ Σ b)  (u/cr)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "global_compare_performance.png")
    plt.close()

    # 👉 N(t) : une seule courbe (même seed pour tous les α)
    plt.figure(figsize=(9, 5))
    first_alpha = next(iter(results))  # prend le premier (peu importe lequel)
    times, _, __, g0 = results[first_alpha]
    plt.plot(times, g0["n_players"])
    plt.title("Nombre de joueurs actifs N(t)")
    plt.xlabel("Temps simulé (secondes)")
    plt.ylabel("N(t) (joueurs)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "global_n_players_single.png")
    plt.close()
    

    print(f"\n✅ Simulation terminée.")
    print(f"Résumé écrit dans : {summary_path}")
    print(f"Journal d’événements écrit dans : {eventlog_path}")


if __name__ == "__main__":
    main()
