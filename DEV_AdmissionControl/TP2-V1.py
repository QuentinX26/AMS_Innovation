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

        departure_time = time + duration
        heapq.heappush(
            self.event_queue,
            TrafficEvent(departure_time, TrafficEvent.DEPARTURE, flow)
        )

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

    def compute_total_utility(self, flow, server):
        j = flow.flow_class
        total = 0.0
        for app in server.get_applications():
            if app.handles(j):
                w = self.compute_w_j_d(j, app)
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
        if self.compute_total_utility(flow, server) < self.U_min:
            return False

        return True




"""
===========================================================
 Construction du système
===========================================================
"""

def build_example_system():
    M = 3
    N = 2

    # Taux d'arrivée ζ_j
    zeta = np.array([3.0, 2.5, 2.0])

    # Taux de service µ_j
    mu = np.array([1.0, 0.8, 0.6])

    # Capacités
    psi_values   = [3, 2]      # compute
    theta_values = [2.5, 1.5]  # access

    # ---- NOUVELLE CONSTRUCTION DE U (équilibrée par capacité) ----
    capacities = np.array(psi_values) + np.array(theta_values)  # C_i
    probs = capacities / capacities.sum()                       # p_i

    # même répartition pour toutes les classes j
    U = np.zeros((N, M))
    for j in range(M):
        U[:, j] = probs

    # Applications
    appA = Application(app_id="A", interested_classes=[0])
    appB = Application(app_id="B", interested_classes=[1])
    appC = Application(app_id="C", interested_classes=[2])

    servers = []
    for i in range(N):
        s = Server(
            server_id=i,
            M=M,
            psi=psi_values[i],
            theta=theta_values[i],
            installed_applications=[appA, appB, appC]
        )
        servers.append(s)

    traffic_gen = AreaTrafficGenerator(
        M=M,
        zeta=zeta,
        mu=mu,
        seed=42
    )

    load_balancer = AreaLoadBalancer(
        M=M,
        N=N,
        routing_matrix=U,
        seed=123
    )

    admission_control = AdmissionControl(
        servers=servers,
        U_min=0.05
    )

    return traffic_gen, load_balancer, servers, admission_control


"""
===========================================================
 Simulation (event-driven)
===========================================================
"""

def run_simulation(T_max=50.0, MAX_EVENTS=1000):

    traffic_gen, load_balancer, servers, admission_control = build_example_system()
    traffic_gen.initialize()

    current_time = 0.0
    event_count = 0

    stats = {"arrivals": 0, "admitted": 0, "rejected": 0, "departures": 0}

    # HISTORIQUE pour les graphiques
    history = {
        "time": [],
        "server_compute": [[] for _ in servers],
        "server_access": [[] for _ in servers],
        "admitted": [],
        "rejected": [],
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

        if current_time > T_max:
            break

        event_count += 1

        if event.type == TrafficEvent.ARRIVAL:

            j = event.flow
            stats["arrivals"] += 1

            flow = traffic_gen.process_arrival(current_time, j)
            server_id = load_balancer.route(flow.flow_class)
            server = servers[server_id]

            if admission_control.accept(flow, server):
                server.admit(flow)
                stats["admitted"] += 1
            else:
                stats["rejected"] += 1

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
    for s in servers:
        print(f"  Serveur {s.id}: Compute={s.compute_load()}  Access={s.access_load:.2f}/{s.theta}")

    return history, servers, stats


"""
===========================================================
 GRAPHIQUES (B)
===========================================================
"""

def plot_server_loads(history, servers):
    time = history["time"]

    for i, s in enumerate(servers):
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

        plt.title(f"Server {s.id} Load Over Time")
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


"""
===========================================================
 MAIN
===========================================================
"""

if __name__ == "__main__":
    history, servers, stats = run_simulation(T_max=50.0, MAX_EVENTS=1000)

    # GRAPH PART
    plot_server_loads(history, servers)
    plot_admissions(history)
