"""
kelly_simulator.py test
------------------
Simulation événementielle du mécanisme de Kelly avec apprentissage par Best-Response (BR)
ou par Gradient Ascent (GA), warm-up, initialisation BR des arrivants, contrôleur de prix
(optionnel), et exports CSV (global + par joueur).

Références conceptuelles:
- Best-Response Learning in Budgeted α-Fair Kelly Mechanisms
- Mécanisme de Kelly (allocation proportionnelle)
"""

import math
import random
import heapq
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple


# ======================================================================
# Utilités α-fair et fonctions auxiliaires
# ======================================================================

def alpha_fair_value(x: float, alpha: float) -> float:
    """V(x) = α-fair utility."""
    if x <= 0:
        return -float("inf")
    if alpha == 1:
        return math.log(x)
    return (x ** (1 - alpha)) / (1 - alpha)


def project_bid(raw_bid: float, epsilon: float, budget: float, price: float) -> float:
    """Projette l’enchère dans l’ensemble faisable [ε/λ, c/λ]."""
    if price <= 0:
        return max(raw_bid, 0.0)
    z_min = epsilon / price
    z_max = budget / price if budget != float("inf") else float("inf")
    return min(max(raw_bid, z_min), z_max)


def best_response_closed_form(alpha: float, a_i: float, s_minus_i: float, lam: float) -> Optional[float]:
    """Best-response analytique pour α ∈ {0,1,2}."""
    s = max(s_minus_i, 0.0)
    if lam <= 0:
        raise ValueError("λ doit être > 0")
    if alpha == 0:
        return max(math.sqrt(max(a_i * s / lam, 0.0)) - s, 0.0)
    elif alpha == 1:
        disc = s * s + 4 * a_i * s / lam
        return max((-s + math.sqrt(max(disc, 0.0))) / 2, 0.0)
    elif alpha == 2:
        return max(math.sqrt(max(a_i * s / lam, 0.0)), 0.0)
    return None


def best_response_generic(alpha: float, a_i: float, s_minus_i: float, lam: float,
                          tol: float = 1e-9, max_iter: int = 200) -> float:
    """Best-response générique (bissection) pour α>0 (au besoin)."""
    s = max(s_minus_i, 0.0)
    if lam <= 0:
        raise ValueError("λ doit être > 0")

    def f(z: float) -> float:
        denom = z + s
        if denom <= 0:
            return -a_i * s / lam
        left = (z / denom) ** alpha if alpha > 0 else 1.0
        right = (a_i * s) / (lam * (denom ** 2))
        return left - right

    if s == 0:
        return 0.0

    z_lo, z_hi = 0.0, 1.0
    f_lo, f_hi = f(z_lo), f(z_hi)
    steps = 0
    while f_lo * f_hi > 0 and steps < 60:
        z_hi *= 2.0
        f_hi = f(z_hi)
        steps += 1
    if f_lo * f_hi > 0:
        # fallback sur forme fermée la plus proche si pas de changement de signe
        cf = best_response_closed_form(round(alpha), a_i, s, lam)
        return cf if cf is not None else 0.0

    for _ in range(max_iter):
        z_mid = 0.5 * (z_lo + z_hi)
        f_mid = f(z_mid)
        if abs(f_mid) < tol or (z_hi - z_lo) < tol:
            return max(z_mid, 0.0)
        if f_lo * f_mid <= 0:
            z_hi, f_hi = z_mid, f_mid
        else:
            z_lo, f_lo = z_mid, f_mid
    return max(0.5 * (z_lo + z_hi), 0.0)


# ======================================================================
# Classes principales : Joueur et Propriétaire
# ======================================================================

@dataclass
class Player:
    id: int
    a: float
    alpha: float
    budget: float
    epsilon: float
    price: float
    bid: float = 0.0
    policy: str = "BR"     # "BR" ou "GA"
    ga_step: float = 0.1
    ga_grad_cap: float = 10.0  # cap du gradient pour GA

    def feasible_bid(self, z: float) -> float:
        return project_bid(z, self.epsilon, self.budget, self.price)

    def allocation(self, total_bids: float, delta: float) -> float:
        denom = total_bids + delta
        return (self.bid / denom) if denom > 0 else 0.0

    def payoff(self, total_bids: float, delta: float) -> float:
        x = self.allocation(total_bids, delta)
        return self.a * alpha_fair_value(x, self.alpha) - self.price * self.bid

    def payoff_grad(self, total_bids: float, delta: float, h: float = 1e-6) -> float:
        """Gradient numérique du payoff pour GA, avec bornes."""
        z0 = self.bid
        bmin, bmax = self.feasible_bid(0.0), self.feasible_bid(float("inf"))
        z_minus, z_plus = max(z0 - h, bmin), min(z0 + h, bmax)
        self.bid = z_minus
        f_minus = self.payoff(total_bids - (z0 - z_minus), delta)
        self.bid = z_plus
        f_plus = self.payoff(total_bids - (z0 - z_plus), delta)
        self.bid = z0
        if z_plus == z_minus:
            return 0.0
        grad = (f_plus - f_minus) / (z_plus - z_minus)
        # cap pour éviter des sauts trop grands
        if grad > self.ga_grad_cap:
            grad = self.ga_grad_cap
        elif grad < -self.ga_grad_cap:
            grad = -self.ga_grad_cap
        return grad

    def best_response(self, s_minus_i: float) -> float:
        cf = best_response_closed_form(self.alpha, self.a, s_minus_i, self.price)
        if cf is None:
            cf = best_response_generic(self.alpha, self.a, s_minus_i, self.price)
        return self.feasible_bid(cf)

    def update_bid(self, s_minus_i: float, total_bids: float, delta: float):
        if self.policy.upper() == "BR":
            self.bid = self.best_response(s_minus_i)
        else:
            grad = self.payoff_grad(total_bids, delta)
            raw = self.bid + self.ga_step * grad
            self.bid = self.feasible_bid(max(raw, 0.0))


@dataclass
class ResourceOwner:
    price: float
    delta: float
    adjust_price_fn: Optional[Callable[['ResourceOwner', List[Player]], float]] = None

    def total_bids(self, players: List[Player]) -> float:
        return sum(p.bid for p in players)

    def allocations(self, players: List[Player]) -> Dict[int, float]:
        T = self.total_bids(players)
        return {p.id: p.bid / (T + self.delta) if (T + self.delta) > 0 else 0.0 for p in players}

    def maybe_adjust_price(self, players: List[Player]):
        if self.adjust_price_fn is not None:
            new_price = self.adjust_price_fn(self, players)
            if new_price and new_price > 0:
                self.price = new_price
                for p in players:
                    p.price = new_price


# ======================================================================
# Simulateur événementiel
# ======================================================================

@dataclass(order=True)
class Event:
    time: float
    kind: str = field(compare=False)
    payload: Optional[dict] = field(default=None, compare=False)


class EventSimulator:
    def __init__(
        self,
        A: float,
        B: float,
        owner: ResourceOwner,
        policy: str = "BR",
        alpha: float = 1.0,
        a_seq: Optional[Callable[[int], float]] = None,
        budget: float = 4000.0,
        epsilon: float = 1e-3,
        max_time: float = 100.0,
        seed: int = 42,
        bid_round_interval: float = 1.0,
        price_update_interval: Optional[float] = None,
        ga_step: float = 0.1,
        ga_grad_cap: float = 10.0,
        warmup_time: float = 0.0,
        init_with_br: bool = False,
        export_players: bool = False,
    ):
        """
        :param A: taux d'arrivée
        :param B: taux de départ (durée de séjour ~ Exp(B))
        """
        self.rng = random.Random(seed)
        self.A = A
        self.B = B
        self.owner = owner
        self.policy = policy
        self.alpha = alpha
        self.a_seq = a_seq if a_seq is not None else (lambda i: 100.0 * (i + 1) ** -0.5)
        self.budget = budget
        self.epsilon = epsilon
        self.max_time = max_time
        self.bid_round_interval = bid_round_interval
        self.price_update_interval = price_update_interval
        self.ga_step = ga_step
        self.ga_grad_cap = ga_grad_cap
        self.warmup_time = warmup_time
        self.init_with_br = init_with_br
        self.export_players = export_players

        self.players: List[Player] = []
        self.next_player_id = 0
        self.event_q: List[Event] = []
        self.time = 0.0

        # Historique
        self.history: List[Dict] = []
        self.per_player_rows: List[Tuple[float, int, float, float]] = []  # (time,id,bid,alloc)

    # --- Outils événements ---
    def exp_time(self, rate: float) -> float:
        return self.rng.expovariate(rate) if rate > 0 else float("inf")

    def schedule(self, t: float, kind: str, payload: Optional[dict] = None):
        heapq.heappush(self.event_q, Event(t, kind, payload))

    # --- initialisation ---
    def initialize(self):
        self.schedule(self.time + self.exp_time(self.A), "ARRIVAL")
        self.schedule(self.time + self.bid_round_interval, "BID_ROUND")
        if self.price_update_interval:
            self.schedule(self.time + self.price_update_interval, "PRICE_UPDATE")

    # --- gestion des joueurs ---
    def add_player(self):
        pid = self.next_player_id
        self.next_player_id += 1
        p = Player(
            id=pid,
            a=self.a_seq(pid),
            alpha=self.alpha,
            budget=self.budget,
            epsilon=self.epsilon,
            price=self.owner.price,
            bid=0.0,
            policy=self.policy,
            ga_step=self.ga_step,
            ga_grad_cap=self.ga_grad_cap,
        )
        # Initialiser par Best-Response si demandé (réduit les phases à z=ε/λ)
        if self.init_with_br:
            total_bids = self.owner.total_bids(self.players)
            s_minus_i = total_bids + self.owner.delta  # somme des autres + delta
            p.bid = p.best_response(s_minus_i - self.owner.delta)
        self.players.append(p)
        self.schedule(self.time + self.exp_time(self.B), "DEPARTURE", {"id": pid})

    def remove_player(self, pid: int):
        self.players = [p for p in self.players if p.id != pid]

    # --- rounds d’enchères ---
    def synchronous_bidding_round(self):
        if not self.players:
            return
        total_bids = self.owner.total_bids(self.players)
        s_minus = {p.id: total_bids - p.bid + self.owner.delta for p in self.players}

        # Calcul BR synchrone
        new_bids: Dict[int, float] = {}
        for p in self.players:
            if p.policy.upper() == "BR":
                new_bids[p.id] = p.best_response(s_minus[p.id] - self.owner.delta)

        # Applique BR
        for p in self.players:
            if p.policy.upper() == "BR":
                p.bid = new_bids[p.id]

        # Met à jour GA (si présent)
        if any(p.policy.upper() != "BR" for p in self.players):
            total_bids = self.owner.total_bids(self.players)
            for p in self.players:
                if p.policy.upper() != "BR":
                    p.update_bid(s_minus[p.id] - self.owner.delta, total_bids, self.owner.delta)

    # --- enregistrement des métriques ---
    def record_metrics(self):
        T = self.owner.total_bids(self.players)
        allocs = self.owner.allocations(self.players)
        snapshot = {
            "time": self.time,
            "n_players": len(self.players),
            "price": self.owner.price,
            "total_bids": T,
            "bids": {p.id: p.bid for p in self.players},
            "allocs": allocs,
            "welfare": sum(p.a * alpha_fair_value(allocs.get(p.id, 0.0), p.alpha)
                           for p in self.players) - self.owner.price * T,
        }
        self.history.append(snapshot)
        if self.export_players:
            for p in self.players:
                self.per_player_rows.append((self.time, p.id, p.bid, allocs.get(p.id, 0.0)))

    # --- boucle principale ---
    def run(self):
        self.initialize()
        while self.event_q and self.time <= self.max_time:
            ev = heapq.heappop(self.event_q)
            self.time = ev.time
            if self.time > self.max_time:
                break

            if ev.kind == "ARRIVAL":
                self.add_player()
                self.schedule(self.time + self.exp_time(self.A), "ARRIVAL")

            elif ev.kind == "DEPARTURE":
                self.remove_player(ev.payload["id"])

            elif ev.kind == "BID_ROUND":
                if self.players:
                    self.synchronous_bidding_round()
                    # Enregistre seulement après warm-up
                    if self.time >= self.warmup_time:
                        self.record_metrics()
                self.schedule(self.time + self.bid_round_interval, "BID_ROUND")

            elif ev.kind == "PRICE_UPDATE":
                self.owner.maybe_adjust_price(self.players)
                if self.price_update_interval:
                    self.schedule(self.time + self.price_update_interval, "PRICE_UPDATE")
        return self.history

    # --- exports ---
    def export_history_csv(self, path: str = "kelly_history.csv"):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "n_players", "price", "total_bids", "welfare"])
            for row in self.history:
                w.writerow([row["time"], row["n_players"], row["price"],
                            row["total_bids"], row["welfare"]])

    def export_per_player_csv(self, path: str = "kelly_bids_allocs.csv"):
        if not self.export_players:
            return
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "player_id", "bid", "alloc"])
            for (t, pid, z, x) in self.per_player_rows:
                w.writerow([t, pid, z, x])


# ======================================================================
# Contrôleur de prix (optionnel)
# ======================================================================

def proportional_price_controller(
    owner: ResourceOwner,
    players: List[Player],
    k: float = 0.02,
    target: float = 50.0,
    min_price: float = 1e-4,
    max_price: float = 1e3,
) -> float:
    """
    Contrôle multiplicatif simple autour d'une somme de bids cible.
    new_price = clamp( price * ( 1 + k * (T-target)/max(target,1e-9) ) ).
    """
    T = owner.total_bids(players)
    factor = 1.0 + k * (T - target) / max(target, 1e-9)
    new_p = owner.price * factor
    if new_p < min_price:
        new_p = min_price
    elif new_p > max_price:
        new_p = max_price
    return new_p


# ======================================================================
# Demo / CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Kelly mechanism event-driven simulator")
    # Processus
    parser.add_argument("--A", type=float, default=0.5, help="Taux d'arrivée")
    parser.add_argument("--B", type=float, default=0.2, help="Taux de départ")
    parser.add_argument("--max-time", type=float, default=50.0, help="Temps max de simulation")
    parser.add_argument("--seed", type=int, default=123, help="Graine aléatoire")
    parser.add_argument("--bid-interval", type=float, default=1.0, help="Intervalle entre rounds d'enchères")
    parser.add_argument("--warmup", type=float, default=0.0, help="Temps de warm-up non enregistré")

    # Mécanisme
    parser.add_argument("--price", type=float, default=1.0, help="Prix initial λ")
    parser.add_argument("--delta", type=float, default=0.1, help="Réservation δ >= 0")
    parser.add_argument("--alpha", type=float, default=1.0, help="Paramètre α-fair (>0)")
    parser.add_argument("--policy", type=str, default="BR", choices=["BR", "GA"], help="Politique d'apprentissage")
    parser.add_argument("--budget", type=float, default=4000.0, help="Budget c_i (identique pour tous)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Paiement minimal ε_i (identique pour tous)")
    parser.add_argument("--ga-step", type=float, default=0.1, help="Pas de GA")
    parser.add_argument("--ga-cap", type=float, default=10.0, help="Cap du gradient pour GA")
    parser.add_argument("--init-br", action="store_true", help="Initialiser chaque nouveau joueur par sa Best-Response")

    # Contrôleur de prix
    parser.add_argument("--price-update", type=float, default=None, help="Période d'update du prix (sec). None = statique")
    parser.add_argument("--price-k", type=float, default=0.02, help="Gain k du contrôleur")
    parser.add_argument("--price-target", type=float, default=50.0, help="Cible de somme de bids pour le contrôleur")
    parser.add_argument("--price-min", type=float, default=1e-4, help="Prix minimal")
    parser.add_argument("--price-max", type=float, default=1e3, help="Prix maximal")

    # Exports
    parser.add_argument("--history-out", type=str, default="kelly_history.csv", help="CSV global (history)")
    parser.add_argument("--export-players", action="store_true", help="Exporter aussi le CSV par joueur")
    parser.add_argument("--players-out", type=str, default="kelly_bids_allocs.csv", help="CSV par joueur")

    args = parser.parse_args()

    # Propriétaire + option de contrôleur
    owner = ResourceOwner(price=args.price, delta=args.delta)
    if args.price_update is not None:
        def _ctrl(o: ResourceOwner, ps: List[Player]):
            return proportional_price_controller(
                o, ps, k=args.price_k, target=args.price_target,
                min_price=args.price_min, max_price=args.price_max
            )
        owner.adjust_price_fn = _ctrl

    # Séquence a_i (ex: décroissante modérée)
    a_seq = lambda i: 100.0 * (i + 1) ** -0.4

    sim = EventSimulator(
        A=args.A, B=args.B, owner=owner, policy=args.policy, alpha=args.alpha,
        a_seq=a_seq, budget=args.budget, epsilon=args.epsilon,
        max_time=args.max_time, seed=args.seed, bid_round_interval=args.bid_interval,
        price_update_interval=args.price_update, ga_step=args.ga_step, ga_grad_cap=args.ga_cap,
        warmup_time=args.warmup, init_with_br=args.init_br, export_players=args.export_players
    )

    history = sim.run()
    sim.export_history_csv(args.history_out)
    if args.export_players:
        sim.export_per_player_csv(args.players_out)

    print(f"[OK] {len(history)} snapshots enregistrés dans {args.history_out}")
    if args.export_players:
        print(f"[OK] Bids/allocs par joueur dans {args.players_out}")


if __name__ == "__main__":
    main()
