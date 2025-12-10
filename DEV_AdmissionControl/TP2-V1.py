import heapq
import math
import numpy as np
import matplotlib.pyplot as plt

"""
===========================================================
 Composant 1 : Area Traffic Generator
===========================================================
"""

class TrafficEvent:
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"

    def __init__(self, time, event_type, flow=None):
        self.time = time
        self.type = event_type
        self.flow = flow

    def __lt__(self, other):
        return self.time < other.time


class Flow:
    def __init__(self, flow_id, flow_class, arrival_time, duration, bitrate):
        self.id = flow_id
        self.flow_class = flow_class
        self.arrival_time = arrival_time
        self.duration = duration
        self.bitrate = bitrate
        self.server_id = None


class AreaTrafficGenerator:

    def __init__(
        self,
        M,
        zeta,
        mu,
        seed=0,
        arrival_dists=None,
        service_dists=None,
        bitrate_dists=None,
    ):
        self.M = M
        self.zeta = np.array(zeta)
        self.mu = np.array(mu)

        self.rng = np.random.default_rng(seed)
        self.event_queue = []
        self.flow_counter = 0

        # Arrivals
        if arrival_dists is None:
            self.arrival_dists = [
                self._exp_dist(1.0 / self.zeta[j])
                for j in range(M)
            ]
        else:
            self.arrival_dists = arrival_dists

        # Service durations
        if service_dists is None:
            self.service_dists = [
                self._exp_dist(1.0 / self.mu[j])
                for j in range(M)
            ]
        else:
            self.service_dists = service_dists

        # Bitrates
        if bitrate_dists is None:
            self.bitrate_dists = [
                (lambda: float(self.rng.lognormal(mean=0.0, sigma=1.0)))
                for _ in range(M)
            ]
        else:
            self.bitrate_dists = bitrate_dists

    def _exp_dist(self, scale):
        def sample():
            return float(self.rng.exponential(scale))
        return sample

    def initialize(self):
        for j in range(self.M):
            t = self.arrival_dists[j]()
            heapq.heappush(
                self.event_queue,
                TrafficEvent(t, TrafficEvent.ARRIVAL, flow=j)
            )

    def next_event(self):
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)

    def process_arrival(self, time, j):
        self.flow_counter += 1

        duration = self.service_dists[j]()
        bitrate = self.bitrate_dists[j]()

        flow = Flow(self.flow_counter, j, time, duration, bitrate)

        next_arrival = time + self.arrival_dists[j]()
        heapq.heappush(
            self.event_queue,
            TrafficEvent(next_arrival, TrafficEvent.ARRIVAL, flow=j)
        )

        return flow


"""
===========================================================
 Composant 2 : Load Balancer
===========================================================
"""

class AreaLoadBalancer:

    def __init__(self, M, N, routing_matrix, seed=0):
        self.M = M
        self.N = N
        self.U = np.array(routing_matrix, dtype=float)
        self.rng = np.random.default_rng(seed)

        for j in range(M):
            if not np.isclose(np.sum(self.U[:, j]), 1.0):
                raise ValueError(f"Colonne {j} de U ne somme pas à 1.")

    def route(self, flow_class):
        probs = self.U[:, flow_class]
        return int(self.rng.choice(self.N, p=probs))


class RoundRobinLoadBalancer(AreaLoadBalancer):
    """Load balancer déterministe en round-robin par classe."""

    def __init__(self, M, N, routing_matrix=None, seed=0):
        matrix = routing_matrix if routing_matrix is not None else (np.ones((N, M)) / N)
        super().__init__(M, N, matrix, seed)
        self.counters = [0] * M

    def route(self, flow_class):
        idx = self.counters[flow_class] % self.N
        self.counters[flow_class] += 1
        return idx


class LeastComputeLoadBalancer(AreaLoadBalancer):
    """Choisit le serveur avec la plus faible charge compute (X)."""

    def __init__(self, M, N, routing_matrix=None, seed=0, servers=None):
        matrix = routing_matrix if routing_matrix is not None else (np.ones((N, M)) / N)
        super().__init__(M, N, matrix, seed)
        if servers is None:
            raise ValueError("servers requis pour LeastComputeLoadBalancer")
        self.servers = servers

    def route(self, flow_class):
        loads = [s.compute_load() for s in self.servers]
        min_load = min(loads)
        candidates = [i for i, l in enumerate(loads) if l == min_load]
        return int(self.rng.choice(candidates))


# Registre pratique pour sélectionner une politique de load balancing par nom
# Descriptions:
# - random (défaut): tirage aléatoire selon la matrice U.
# - round_robin: rotation déterministe par classe.
# - least_compute: envoie vers le serveur avec la plus faible charge compute.
LOAD_BALANCER_POLICIES = {
    "random": AreaLoadBalancer,
    "round_robin": RoundRobinLoadBalancer,
    "least_compute": LeastComputeLoadBalancer,
}


"""
===========================================================
 Composant 3 : Server
===========================================================
"""

class Server:

    def __init__(self, server_id, M, psi, theta, installed_applications):
        self.id = server_id
        self.M = M
        self.psi = psi
        self.theta = theta
        self.Di = installed_applications

        self.X = [0] * M
        self.access_load = 0.0
        self.active_flows = {}

    def compute_load(self):
        return sum(self.X)

    def has_compute_capacity(self):
        return self.compute_load() < self.psi

    def has_access_capacity(self):
        return self.access_load < self.theta

    def admit(self, flow):
        j = flow.flow_class
        self.X[j] += 1
        self.access_load += flow.bitrate

        if abs(self.access_load) < 1e-12:
            self.access_load = 0.0

        flow.server_id = self.id
        self.active_flows[flow.id] = flow

    def remove(self, flow):
        j = flow.flow_class
        if flow.id in self.active_flows:
            self.X[j] -= 1
            self.access_load -= flow.bitrate

            if abs(self.access_load) < 1e-12:
                self.access_load = 0.0

            del self.active_flows[flow.id]

    def get_applications(self):
        return self.Di


"""
===========================================================
 Composant 5 : Applications & Utilité
===========================================================
"""

class Application:

    def __init__(self, app_id, interested_classes, a=1.0, b=0.2):
        self.id = app_id
        self.interested_classes = set(interested_classes)
        self.a = a
        self.b = b

    def handles(self, flow_class):
        return flow_class in self.interested_classes

    def compute_utility(self, flow_class, current_count):
        if not self.handles(flow_class):
            return 0.0
        return self.a * math.exp(-self.b * current_count)


"""
===========================================================
 Composant 4 : Admission Control
===========================================================
"""

class AdmissionControl:

    def __init__(self, servers, U_min=0.05):
        self.servers = servers
        self.U_min = U_min

    def compute_w_j_d(self, flow_class, app):
        count = 0
        for s in self.servers:
            for f in s.active_flows.values():
                if f.flow_class == flow_class and app.handles(flow_class):
                    count += 1
        return count

    def compute_total_utility(self, flow, server, include_new_flow=True):
        j = flow.flow_class
        total = 0.0
        for app in server.get_applications():
            if app.handles(j):
                w = self.compute_w_j_d(j, app)
                if include_new_flow:
                    w += 1
                total += app.compute_utility(j, w)
        return total

    def access_cost(self, server):
        return server.access_load / server.theta



    def accept(self, flow, server):
        # Capacité compute (prospective : +1 flux)
        if server.compute_load() + 1 > server.psi:
            return False

        # Capacité accès (prospective : +bitrate)
        future_access = server.access_load + flow.bitrate
        if future_access > server.theta:
            return False

        # Utilité
        if self.compute_total_utility(flow, server, include_new_flow=True) < self.U_min:
            return False

        return True


class CapacityOnlyAdmission(AdmissionControl):
    """
    Variante : contrôle d'admission qui ne regarde que les capacités compute et accès.
    Utilité ignorée.
    """

    def __init__(self, servers, **kwargs):
        super().__init__(servers, U_min=0.0)

    def accept(self, flow, server):
        if server.compute_load() + 1 > server.psi:
            return False
        if server.access_load + flow.bitrate > server.theta:
            return False
        return True


class UtilityGateAdmission(AdmissionControl):
    """
    Variante : contrôle d'admission qui vérifie utilité mais
    ne considère pas la charge accès (seulement compute + utilité).
    """

    def accept(self, flow, server):
        if server.compute_load() + 1 > server.psi:
            return False
        # Ignore l'accès pour tester la sensibilité utilité seule
        if self.compute_total_utility(flow, server, include_new_flow=True) < self.U_min:
            return False
        return True


class RandomizedAdmission(AdmissionControl):
    """
    Variante : admission probabiliste après vérification des capacités et utilité.
    """

    def __init__(self, servers, U_min=0.05, p_accept=0.5, seed=0):
        super().__init__(servers, U_min=U_min)
        self.p_accept = p_accept
        self.rng = np.random.default_rng(seed)

    def accept(self, flow, server):
        if server.compute_load() + 1 > server.psi:
            return False
        if server.access_load + flow.bitrate > server.theta:
            return False
        if self.compute_total_utility(flow, server, include_new_flow=True) < self.U_min:
            return False
        return bool(self.rng.random() < self.p_accept)


# Registre pratique pour sélectionner une politique par nom
# Descriptions:
# - baseline: contrôle compute + accès + utilité (avec seuil U_min).
# - capacity_only: contrôle compute + accès uniquement, ignore l'utilité.
# - utility_gate: contrôle compute + utilité uniquement, ignore l'accès.
# - randomized: contrôle compute + accès + utilité puis décide aléatoirement avec p_accept.
ADMISSION_POLICIES = {
    "baseline": AdmissionControl,
    "capacity_only": CapacityOnlyAdmission,
    "utility_gate": UtilityGateAdmission,
    "randomized": RandomizedAdmission,
}




"""
===========================================================
 Construction du système
===========================================================
"""

def build_example_system(
    M=3,
    N=2,
    zeta=None,
    mu=None,
    psi_values=None,
    theta_values=None,
    routing_matrix=None,
    lb_policy=None,
    lb_kwargs=None,
    arrival_dists=None,
    service_dists=None,
    bitrate_dists=None,
    seed_traffic=42,
    seed_lb=123,
    admission_control_factory=None,
    admission_policy=None,
    admission_kwargs=None,
    applications=None,
):
    """Build a system instance with fully parametric inputs."""
    zeta = np.array(zeta if zeta is not None else [3.0, 2.5, 2.0], dtype=float)
    mu = np.array(mu if mu is not None else [1.0, 0.8, 0.6], dtype=float)

    psi_values = psi_values if psi_values is not None else [3, 2]
    theta_values = theta_values if theta_values is not None else [2.5, 1.5]

    admission_kwargs = admission_kwargs or {}

    # Routing matrix U: if not provided, distribute according to capacity weights.
    if routing_matrix is None:
        capacities = np.array(psi_values) + np.array(theta_values)
        probs = capacities / capacities.sum()
        U = np.zeros((N, M))
        for j in range(M):
            U[:, j] = probs
    else:
        U = np.array(routing_matrix, dtype=float)

    # Applications: default = one app per classe, labellées A, B, C...
    if applications is None:
        app_ids = [chr(ord("A") + j) for j in range(M)] if M <= 26 else [f"app-{j}" for j in range(M)]
        applications = [Application(app_id=app_ids[j], interested_classes=[j]) for j in range(M)]

    lb_kwargs = lb_kwargs or {}
    servers = []
    for i in range(N):
        s = Server(
            server_id=i,
            M=M,
            psi=psi_values[i],
            theta=theta_values[i],
            installed_applications=applications,
        )
        servers.append(s)

    traffic_gen = AreaTrafficGenerator(
        M=M,
        zeta=zeta,
        mu=mu,
        seed=seed_traffic,
        arrival_dists=arrival_dists,
        service_dists=service_dists,
        bitrate_dists=bitrate_dists,
    )

    # Load balancer selection
    if lb_policy:
        lb_factory = LOAD_BALANCER_POLICIES.get(lb_policy)
        if lb_factory is None:
            raise ValueError(f"Load balancer policy '{lb_policy}' inconnue.")
    else:
        lb_factory = AreaLoadBalancer

    # Ensure servers are available for policies that need them
    if lb_factory is LeastComputeLoadBalancer and "servers" not in lb_kwargs:
        lb_kwargs["servers"] = servers

    load_balancer = lb_factory(
        M=M,
        N=N,
        routing_matrix=U,
        seed=seed_lb,
        **lb_kwargs,
    )

    if admission_control_factory is None and admission_policy:
        factory = ADMISSION_POLICIES.get(admission_policy)
        if factory is None:
            raise ValueError(f"Admission policy '{admission_policy}' inconnue.")
    else:
        factory = admission_control_factory or AdmissionControl

    admission_control = factory(servers=servers, **admission_kwargs)

    return traffic_gen, load_balancer, servers, admission_control


"""
===========================================================
 Simulation (event-driven)
===========================================================
"""

def run_simulation(T_max=50.0, MAX_EVENTS=1000, **build_kwargs):
    """
    Run an event-driven simulation.
    build_kwargs are forwarded to build_example_system to allow full parametrization
    (M, N, capacities, routing matrix, seeds, distributions, admission policy, etc.).
    """

    traffic_gen, load_balancer, servers, admission_control = build_example_system(**build_kwargs)
    traffic_gen.initialize()

    current_time = 0.0
    event_count = 0

    # Noms de classes (optionnel: passer class_names=["A","B",...])
    class_names = build_kwargs.get("class_names")
    M_classes = servers[0].M if servers else 0
    if class_names is None:
        letters = [chr(ord("A") + j) for j in range(min(26, M_classes))]
        class_names = letters if M_classes <= 26 else [f"Class {j}" for j in range(M_classes)]

    # Noms d'applications (pour graphiques par appli)
    app_names = [app.id for app in (servers[0].get_applications() if servers else [])]

    stats = {"arrivals": 0, "admitted": 0, "rejected": 0, "departures": 0}
    class_stats = {
        "arrivals": [0] * M_classes,
        "admitted": [0] * M_classes,
        "rejected": [0] * M_classes,
    }
    app_stats = {app_id: 0 for app_id in app_names}

    # HISTORIQUE pour les graphiques
    history = {
        "time": [],
        "server_compute": [[] for _ in servers],
        "server_access": [[] for _ in servers],
        "admitted": [],
        "rejected": [],
        "class_admitted": [[] for _ in range(M_classes)],
        "class_rejected": [[] for _ in range(M_classes)],
        "class_arrivals": [[] for _ in range(M_classes)],
        "class_names": class_names,
        "app_names": app_names,
        "app_admitted": {app_id: [] for app_id in app_names},
    }

    while event_count < MAX_EVENTS:
        event = traffic_gen.next_event()
        if event is None:
            break

        current_time = event.time

        # SAVE STATE
        history["time"].append(current_time)
        for i, s in enumerate(servers):
            history["server_compute"][i].append(s.compute_load())
            history["server_access"][i].append(s.access_load)
        history["admitted"].append(stats["admitted"])
        history["rejected"].append(stats["rejected"])
        for j in range(M_classes):
            history["class_admitted"][j].append(class_stats["admitted"][j])
            history["class_rejected"][j].append(class_stats["rejected"][j])
            history["class_arrivals"][j].append(class_stats["arrivals"][j])
        for app_id in app_names:
            history["app_admitted"][app_id].append(app_stats[app_id])

        if current_time > T_max:
            break

        event_count += 1

        if event.type == TrafficEvent.ARRIVAL:

            j = event.flow
            stats["arrivals"] += 1
            class_stats["arrivals"][j] += 1

            flow = traffic_gen.process_arrival(current_time, j)
            server_id = load_balancer.route(flow.flow_class)
            server = servers[server_id]

            if admission_control.accept(flow, server):
                server.admit(flow)
                departure_time = current_time + flow.duration
                heapq.heappush(
                    traffic_gen.event_queue,
                    TrafficEvent(departure_time, TrafficEvent.DEPARTURE, flow)
                )
                stats["admitted"] += 1
                class_stats["admitted"][j] += 1
                for app in server.get_applications():
                    if app.handles(j):
                        app_stats[app.id] += 1
            else:
                stats["rejected"] += 1
                class_stats["rejected"][j] += 1

        elif event.type == TrafficEvent.DEPARTURE:

            flow = event.flow
            stats["departures"] += 1

            if flow.server_id is not None:
                servers[flow.server_id].remove(flow)

    print("=== FIN DE SIMULATION ===")
    print(f"Temps simulé : {current_time:.3f}")
    print(f"Événements traités : {event_count}")
    print(f"Arrivées : {stats['arrivals']}")
    print(f"Admis    : {stats['admitted']}")
    print(f"Rejetés  : {stats['rejected']}")
    print(f"Départs  : {stats['departures']}")

    print("\nÉtat final des serveurs :")
    for display_id, s in enumerate(servers, start=1):
        print(f"  Serveur {display_id}: Compute={s.compute_load()}  Access={s.access_load:.2f}/{s.theta}")

    return history, servers, stats


"""
===========================================================
 GRAPHIQUES (B)
===========================================================
"""

def plot_server_loads(history, servers):
    time = history["time"]
    if not time:
        print("No events recorded; skipping server load plots.")
        return

    for i, s in enumerate(servers):
        display_id = i + 1
        plt.figure(figsize=(10,4))

        # Compute load
        plt.step(
            time,
            history["server_compute"][i],
            where="post",
            linewidth=2,
            label="Compute Load"
        )

        # Access load
        plt.step(
            time,
            history["server_access"][i],
            where="post",
            linewidth=2,
            label="Access Load"
        )

        # Threshold
        plt.hlines(
            s.theta,
            xmin=time[0],
            xmax=time[-1],
            colors="r",
            linestyles="--",
            linewidth=1.5,
            label=f"Access Capacity θ={s.theta}"
        )

        plt.title(f"Server {display_id} Load Over Time")
        plt.xlabel("Time")
        plt.ylabel("Load")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()





def plot_admissions(history):
    time = history["time"]
    plt.figure(figsize=(10,5))
    plt.plot(time, history["admitted"], label="Admitted", color="green")
    plt.plot(time, history["rejected"], label="Rejected", color="red")
    plt.legend(); plt.grid()
    plt.xlabel("Time"); plt.ylabel("Cumulative count")
    plt.title("Admissions vs Rejections")
    plt.show()


def plot_class_admissions(history):
    time = history["time"]
    if not time or "class_admitted" not in history:
        print("No class-level data to plot.")
        return

    num_classes = len(history["class_admitted"])
    names = history.get("class_names") or [f"Class {j}" for j in range(num_classes)]
    plt.figure(figsize=(10,5))
    for j in range(num_classes):
        plt.step(time, history["class_admitted"][j], where="post", label=f"{names[j]} Admitted")
    plt.xlabel("Time")
    plt.ylabel("Cumulative admitted")
    plt.title("Admissions par classe")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    for j in range(num_classes):
        plt.step(time, history["class_rejected"][j], where="post", label=f"{names[j]} Rejected")
    plt.xlabel("Time")
    plt.ylabel("Cumulative rejected")
    plt.title("Rejets par classe")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    for j in range(num_classes):
        plt.step(time, history["class_arrivals"][j], where="post", label=f"{names[j]} Arrivals")
    plt.xlabel("Time")
    plt.ylabel("Cumulative arrivals")
    plt.title("Arrivées par classe")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_application_admissions(history):
    time = history["time"]
    app_admitted = history.get("app_admitted")
    app_names = history.get("app_names", [])
    if not time or not app_admitted:
        print("No application-level data to plot.")
        return

    plt.figure(figsize=(10,5))
    for app_id in app_names:
        plt.step(time, app_admitted[app_id], where="post", label=f"App {app_id}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative admitted (by applications)")
    plt.title("Admissions par application")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_area_totals(history):
    """Bar chart des totaux par area (classe) : arrivées, admis, rejetés, taux d'acceptation."""
    class_adm = history.get("class_admitted")
    class_rej = history.get("class_rejected")
    class_arr = history.get("class_arrivals")
    names = history.get("class_names")
    if not class_adm or not names:
        print("No area-level data to plot.")
        return

    admitted = [serie[-1] if serie else 0 for serie in class_adm]
    rejected = [serie[-1] if serie else 0 for serie in class_rej]
    arrivals = [serie[-1] if serie else 0 for serie in class_arr]
    acceptance = []
    for a, r in zip(admitted, rejected):
        total = a + r
        acceptance.append((a / total) if total > 0 else 0.0)

    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(10,5))
    plt.bar(x - width, arrivals, width, label="Arrivals", color="gray")
    plt.bar(x, admitted, width, label="Admitted", color="green")
    plt.bar(x + width, rejected, width, label="Rejected", color="red")
    plt.xticks(x, names)
    plt.ylabel("Count")
    plt.title("Totals par area (classe)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.bar(names, acceptance, color="blue")
    plt.ylabel("Acceptance rate")
    plt.title("Taux d'acceptation par area")
    plt.tight_layout()
    plt.show()

def compare_admission_policies(
    policies,
    T_max=50.0,
    MAX_EVENTS=1000,
    lb_policy="random",
    lb_kwargs=None,
    base_build_kwargs=None,
):
    """
    Compare plusieurs politiques d'admission et trace un bar chart (admis/rejetés).
    policies: liste de noms (dans ADMISSION_POLICIES) ou de factories.
    base_build_kwargs: kwargs communs passés à run_simulation (zeta, mu, etc.).
    """
    lb_kwargs = lb_kwargs or {}
    base_build_kwargs = base_build_kwargs or {}

    labels = []
    admitted = []
    rejected = []
    acceptance_rate = []
    avg_compute = []
    avg_access = []

    for policy in policies:
        # Préparer les kwargs par run
        build_kwargs = dict(
            admission_policy=None if callable(policy) else policy,
            admission_control_factory=policy if callable(policy) else None,
            lb_policy=lb_policy,
            lb_kwargs=lb_kwargs,
            **base_build_kwargs,
        )
        history, servers, stats = run_simulation(
            T_max=T_max,
            MAX_EVENTS=MAX_EVENTS,
            **build_kwargs,
        )
        label = policy.__name__ if callable(policy) else str(policy)
        labels.append(label)
        admitted.append(stats["admitted"])
        rejected.append(stats["rejected"])
        total = stats["admitted"] + stats["rejected"]
        acceptance_rate.append((stats["admitted"] / total) if total > 0 else 0.0)
        # Moyennes de charge compute et accès sur toute la simulation
        all_compute = [val for serie in history["server_compute"] for val in serie]
        all_access = [val for serie in history["server_access"] for val in serie]
        avg_compute.append(float(np.mean(all_compute)) if all_compute else 0.0)
        avg_access.append(float(np.mean(all_access)) if all_access else 0.0)

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, admitted, width, label="Admitted", color="green")
    plt.bar(x + width/2, rejected, width, label="Rejected", color="red")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title("Comparaison des politiques d'admission")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure supplémentaire: taux d'acceptation et charges moyennes
    width2 = 0.25
    plt.figure(figsize=(10,5))
    plt.bar(x - width2, acceptance_rate, width2, label="Acceptance rate", color="blue")
    plt.bar(x, avg_compute, width2, label="Avg compute load", color="orange")
    plt.bar(x + width2, avg_access, width2, label="Avg access load", color="purple")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Taux d'acceptation et charges moyennes (admission)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_load_balancers(
    policies,
    T_max=50.0,
    MAX_EVENTS=1000,
    admission_policy="baseline",
    admission_kwargs=None,
    base_build_kwargs=None,
):
    """
    Compare plusieurs politiques de load balancing et trace un bar chart (admis/rejetés).
    policies: liste de noms (dans LOAD_BALANCER_POLICIES) ou de factories.
    """
    admission_kwargs = admission_kwargs or {}
    base_build_kwargs = base_build_kwargs or {}

    labels = []
    admitted = []
    rejected = []
    acceptance_rate = []
    avg_compute = []
    avg_access = []

    for policy in policies:
        build_kwargs = dict(
            lb_policy=None if callable(policy) else policy,
            lb_kwargs={},
            admission_policy=admission_policy,
            admission_kwargs=admission_kwargs,
            **base_build_kwargs,
        )
        history, servers, stats = run_simulation(
            T_max=T_max,
            MAX_EVENTS=MAX_EVENTS,
            **build_kwargs,
        )
        label = policy.__name__ if callable(policy) else str(policy)
        labels.append(label)
        admitted.append(stats["admitted"])
        rejected.append(stats["rejected"])
        total = stats["admitted"] + stats["rejected"]
        acceptance_rate.append((stats["admitted"] / total) if total > 0 else 0.0)
        all_compute = [val for serie in history["server_compute"] for val in serie]
        all_access = [val for serie in history["server_access"] for val in serie]
        avg_compute.append(float(np.mean(all_compute)) if all_compute else 0.0)
        avg_access.append(float(np.mean(all_access)) if all_access else 0.0)

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, admitted, width, label="Admitted", color="green")
    plt.bar(x + width/2, rejected, width, label="Rejected", color="red")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title("Comparaison des politiques de load balancing")
    plt.legend()
    plt.tight_layout()
    plt.show()

    width2 = 0.25
    plt.figure(figsize=(10,5))
    plt.bar(x - width2, acceptance_rate, width2, label="Acceptance rate", color="blue")
    plt.bar(x, avg_compute, width2, label="Avg compute load", color="orange")
    plt.bar(x + width2, avg_access, width2, label="Avg access load", color="purple")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Taux d'acceptation et charges moyennes (load balancing)")
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
===========================================================
 MAIN
===========================================================
"""

if __name__ == "__main__":
    """
    Politiques d'admission disponibles :
    "baseline": AdmissionControl           : vérifie compute (<ψ), accès (<θ) et utilité ≥ U_min (utilité prospective inclut le flux).
    "capacity_only": CapacityOnlyAdmission : vérifie compute (<ψ) et accès (<θ), ignore l’utilité.
    "utility_gate": UtilityGateAdmission   : vérifie compute (<ψ) et utilité ≥ U_min, ignore l’accès.
    "randomized": RandomizedAdmission      : vérifie compute (<ψ), accès (<θ), utilité ≥ U_min puis accepte avec proba p_accept.
    """
    """
    Politiques de load balancing disponibles :
    "random"       : tirage aléatoire selon la matrice U.
    "round_robin"  : rotation déterministe par classe.
    "least_compute": envoie vers le serveur avec la plus faible charge compute.
    """
    # Choisir la politique d'admission via une variable unique
    ADMISSION_POLICY = "baseline" 
    ADMISSION_KWARGS = {}          # ex: {"p_accept": 0.7, "seed": 123} pour randomized
    LOAD_BALANCER_POLICY = "random"  # "round_robin", "least_compute"
    LOAD_BALANCER_KWARGS = {}        # ex: {} ou {"servers": ...} pour policies spécifiques
    COMPARE_POLICIES = False        # Passe à True pour comparer plusieurs politiques
    COMPARE_LB = False            # Passe à True pour comparer plusieurs load balancers

    if COMPARE_POLICIES:
        # Exemple de liste à comparer
        policies_to_compare = ["baseline", "capacity_only", "utility_gate", "randomized"]
        compare_admission_policies(
            policies=policies_to_compare,
            T_max=50.0,
            MAX_EVENTS=1000,
            lb_policy=LOAD_BALANCER_POLICY,
            lb_kwargs=LOAD_BALANCER_KWARGS,
            base_build_kwargs={"admission_kwargs": ADMISSION_KWARGS},
        )
    elif COMPARE_LB:
        lb_to_compare = ["random", "round_robin", "least_compute"]
        compare_load_balancers(
            policies=lb_to_compare,
            T_max=50.0,
            MAX_EVENTS=1000,
            admission_policy=ADMISSION_POLICY,
            admission_kwargs=ADMISSION_KWARGS,
            base_build_kwargs={},
        )
    else:
        history, servers, stats = run_simulation(
            T_max=50.0,
            MAX_EVENTS=1000,
            admission_policy=ADMISSION_POLICY,
            admission_kwargs=ADMISSION_KWARGS,
            lb_policy=LOAD_BALANCER_POLICY,
            lb_kwargs=LOAD_BALANCER_KWARGS,
        )

        # GRAPH PART
        plot_server_loads(history, servers)
        plot_admissions(history)
        plot_class_admissions(history)
        plot_application_admissions(history)
        plot_area_totals(history)
