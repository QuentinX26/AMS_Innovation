from dataclasses import dataclass, field
import heapq
from typing import Callable, Any, Dict, List, Optional

# ---------------------------------------------------------------
# Classe représentant un événement planifié dans la simulation
# ---------------------------------------------------------------
@dataclass(order=True)
class Event:
    time: float                   # Temps de déclenchement de l'événement
    priority: int                 # Priorité en cas d'événements simultanés
    type: str = field(compare=False)            # Type (ex: ARRIVAL, DEPARTURE, BID_UPDATE)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)  # Données associées


# ---------------------------------------------------------------
# Moteur de simulation événementielle
# ---------------------------------------------------------------
class Simulator:
    """
    Ce simulateur est un moteur à événements discrets :
    - Chaque événement est planifié à un instant donné
    - Les événements sont stockés dans une file triée par temps (min-heap)
    - Lorsqu’un événement est traité, il peut en planifier d’autres
    - Le temps de simulation avance en “sautant” d’un événement à l’autre
    """

    def __init__(self, seed: Optional[int] = 42):
        import random
        self.time: float = 0.0               # Horloge actuelle de la simulation
        self._queue: List[Event] = []        # File de priorité contenant les événements futurs
        self._handlers: Dict[str, Callable[[Event], None]] = {}  # Map type → fonction à exécuter
        self.random = random.Random(seed)    # Générateur pseudo-aléatoire (reproductible)
        self.running = False                 # Flag indiquant si la simulation est active

    # ------------------------------------------------------------------
    # Décorateur pour enregistrer une fonction comme gestionnaire d’un type d’événement
    # Exemple :
    #   @sim.on("ARRIVAL")
    #   def arrival_handler(evt): ...
    # ------------------------------------------------------------------
    def on(self, event_type: str):
        def decorator(func: Callable[[Event], None]):
            self._handlers[event_type] = func
            return func
        return decorator

    # ------------------------------------------------------------------
    # Planifie un événement futur dans la file d’attente
    # ------------------------------------------------------------------
    def schedule(self, time: float, type_: str, priority: int = 0, **payload):
        """
        Planifie un nouvel événement à un instant donné.

        Args:
            time (float): instant de déclenchement
            type_ (str): nom du type d’événement
            priority (int): priorité relative (événements plus prioritaires passent avant)
            payload (dict): données optionnelles passées au handler
        """
        if time < self.time:
            raise ValueError("Un événement ne peut pas être planifié dans le passé.")
        heapq.heappush(self._queue, Event(time, priority, type_, payload))

    # ------------------------------------------------------------------
    # Boucle principale : exécute les événements dans l’ordre chronologique
    # ------------------------------------------------------------------
    def run(self, until_time: Optional[float] = None, max_events: Optional[int] = None):
        """
        Fait tourner la simulation jusqu’à :
          - atteindre un temps maximal (until_time)
          - ou exécuter un certain nombre d’événements (max_events)

        La simulation s’arrête quand la file d’événements est vide ou que la condition est remplie.
        """
        self.running = True
        processed = 0

        while self.running and self._queue:
            evt = heapq.heappop(self._queue)     # Récupère le prochain événement (le plus ancien dans le temps)

            # Si on dépasse la limite temporelle, on remet l’événement dans la file
            if until_time is not None and evt.time > until_time:
                heapq.heappush(self._queue, evt)
                break

            # On avance le temps global à celui de l’événement
            self.time = evt.time

            # On cherche le gestionnaire associé à ce type d’événement
            handler = self._handlers.get(evt.type)
            if handler is None:
                raise KeyError(f"Aucun handler pour l'événement '{evt.type}'")

            # Exécution de la logique de l’événement
            handler(evt)

            processed += 1

            # Si on atteint une limite de nombre d’événements, on s’arrête
            if max_events is not None and processed >= max_events:
                break

    # ------------------------------------------------------------------
    # Arrête la simulation proprement
    # ------------------------------------------------------------------
    def stop(self):
        """Stoppe la simulation (ex: via une condition externe)"""
        self.running = False
