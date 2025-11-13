from dataclasses import dataclass, field
from typing import Dict, List
from kelly import kelly_allocation, best_response_closed_form, project_bid

# ------------------------------------------------------------------
# Représentation d’un joueur (agent économique)
# ------------------------------------------------------------------
@dataclass
class Player:
    id: int
    a: float             # Poids ou efficacité du joueur (préférence pour la ressource)
    budget: float        # Budget maximum disponible
    eps: float           # Petitesse >0 pour éviter divisions par zéro
    lam: float           # Coût marginal (λ)
    alpha: int           # Paramètre d’équité (définit la fonction d’utilité)
    bid: float = 0.0     # Enchère courante du joueur
    alive: bool = True   # Si le joueur est actif dans le système

    @property
    def max_bid(self) -> float:
        """Enchère maximale autorisée (limite de budget)"""
        return self.budget / self.lam if self.lam > 0 else float("inf")

    @property
    def min_bid(self) -> float:
        """Enchère minimale (liée à epsilon pour stabilité numérique)"""
        return self.eps / self.lam if self.lam > 0 else 0.0


# ------------------------------------------------------------------
# Représentation du propriétaire de la ressource
# ------------------------------------------------------------------
@dataclass
class ResourceOwner:
    C: float = 1.0       # Capacité totale disponible
    delta: float = 0.0   # Facteur de stabilisation du dénominateur

    def allocate(self, bids: Dict[int, float]) -> Dict[int, float]:
        """Retourne une allocation proportionnelle (règle de Kelly)"""
        return kelly_allocation(bids, self.C, self.delta)


# ------------------------------------------------------------------
# Classe d’enregistrement des métriques globales de la simulation
# ------------------------------------------------------------------
@dataclass
class Metrics:
    times: List[float] = field(default_factory=list)
    sum_bids: List[float] = field(default_factory=list)
    price: List[float] = field(default_factory=list)
    n_players: List[int] = field(default_factory=list)

    def record(self, t: float, bids: Dict[int, float], C: float):
        """
        Enregistre l’état global à l’instant t :
        - somme totale des enchères Σ b_i
        - prix implicite p = Σ b / C
        - nombre de joueurs actifs
        """
        s = sum(bids.values())
        self.times.append(t)
        self.sum_bids.append(s)
        self.price.append(s / C if C > 0 else 0.0)
        self.n_players.append(len(bids))


# ------------------------------------------------------------------
# Fonction principale : meilleure réponse synchrone
# ------------------------------------------------------------------
def synchronous_best_response(players: Dict[int, Player], owner: ResourceOwner):
    """
    Calcule simultanément la nouvelle enchère de chaque joueur.
    Étapes :
      1. On calcule la somme totale actuelle des enchères Σ b_i
      2. Pour chaque joueur i :
         - On calcule s_minus_i = Σ_{j≠i} b_j
         - On déduit la meilleure réponse b_i*
         - On la projette dans les bornes autorisées
      3. On met à jour toutes les enchères en même temps
    """
    total = sum(p.bid for p in players.values() if p.alive)
    new_bids = {}

    for i, p in players.items():
        if not p.alive:
            continue

        # Somme des autres enchères
        s_minus = (total - p.bid) + owner.delta

        # Calcul analytique du best-response selon α
        z_tilde = best_response_closed_form(p.alpha, p.a, s_minus, p.lam)

        # Projection pour rester dans les bornes
        z_proj = project_bid(z_tilde, p.min_bid, p.max_bid)

        new_bids[i] = z_proj

    # Mise à jour simultanée
    for i, z in new_bids.items():
        players[i].bid = z


# ------------------------------------------------------------------
# Extraction des enchères des joueurs actifs uniquement
# ------------------------------------------------------------------
def alive_bids(players: Dict[int, Player]) -> Dict[int, float]:
    """Retourne le dictionnaire des enchères des joueurs actuellement vivants"""
    return {i: p.bid for i, p in players.items() if p.alive}
