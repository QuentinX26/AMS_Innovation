from typing import Dict
import math

# ------------------------------------------------------------------
# Fonction d’allocation Kelly (proportionnelle aux enchères)
# ------------------------------------------------------------------
def kelly_allocation(bids: Dict[int, float], C: float = 1.0, delta: float = 0.0) -> Dict[int, float]:
    """
    Calcule l’allocation de ressource selon le mécanisme de Kelly :
        x_i = C * (b_i / (Σ b_j + δ))
    Cette formule garantit :
      - Efficacité (pas de gaspillage de ressources)
      - Fairness (allocation proportionnelle)
    """
    total = sum(bids.values()) + float(delta)  # somme des enchères + correction δ

    if total <= 0.0:
        # Cas limite : personne ne mise → personne ne reçoit
        return {i: 0.0 for i in bids}

    # Allocation proportionnelle : chaque joueur reçoit une part de C selon sa mise
    return {i: C * (bids[i] / total) for i in bids}


# ------------------------------------------------------------------
# Meilleure réponse fermée (best-response) pour un joueur donné
# ------------------------------------------------------------------
def best_response_closed_form(alpha: int, a_i: float, s_minus_i: float, lam: float) -> float:
    """
    Calcule analytiquement la meilleure enchère b_i* du joueur i,
    en supposant que les autres enchères sont fixées (Σ_{j≠i} b_j = s_minus_i).

    Paramètres :
        alpha : paramètre d’équité (0 → utilitaire pur, 1 → logarithmique, 2 → fairness)
        a_i   : coefficient d’utilité du joueur i
        s_minus_i : somme des autres enchères
        lam   : facteur de pénalisation des coûts

    Mathématiquement :
        Chaque joueur maximise : U_i(x_i) - λ * b_i
        avec x_i = C * b_i / (Σ b + δ)
    """

    s = max(0.0, float(s_minus_i))  # sécurité numérique : on évite les négatifs

    if alpha == 0:
        # Cas α = 0 → U(x) = x → best response : sqrt(a_i*s/λ) - s
        val = math.sqrt(max(0.0, a_i * s / lam)) - s
        return max(0.0, val)

    elif alpha == 1:
        # Cas α = 1 → U(x) = log(x) → best response issue d’une équation quadratique
        disc = s * s + 4.0 * a_i * s / lam  # discriminant
        val = (-s + math.sqrt(max(0.0, disc))) / 2.0
        return max(0.0, val)

    elif alpha == 2:
        # Cas α = 2 → U(x) = -1/x (équité proportionnelle)
        return math.sqrt(max(0.0, a_i * s / lam))

    else:
        raise ValueError("alpha doit être 0, 1 ou 2.")


# ------------------------------------------------------------------
# Projection d'une enchère dans les bornes autorisées
# ------------------------------------------------------------------
def project_bid(z_tilde: float, eps_over_lam: float, c_over_lam: float) -> float:
    """
    Contrainte sur les enchères pour éviter des valeurs irréalistes :
      - bornes min : ε/λ
      - bornes max : C/λ

    Cela garantit que b_i reste dans [min_bid, max_bid].
    """
    low = max(0.0, float(eps_over_lam))
    high = max(low, float(c_over_lam))
    return min(max(z_tilde, low), high)
