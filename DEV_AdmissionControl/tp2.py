import numpy as np
import matplotlib.pyplot as plt
import csv
import os


# ============================================================
#  COMPONENT 1 — TRAFFIC GENERATOR
# ============================================================

class TrafficGenerator:
    def __init__(self, M, lambdas, mu, bitrate_dist=None):
        self.M = M
        self.lambdas = lambdas
        self.mu = mu
        self.bitrate_dist = bitrate_dist if bitrate_dist else lambda: 1

    def next_arrival(self):
        inter = [np.random.exponential(1/l) for l in self.lambdas]
        j = np.argmin(inter)
        return inter[j], j

    def flow_duration(self, j):
        return np.random.exponential(1/self.mu[j])

    def bitrate(self):
        return self.bitrate_dist()


# ============================================================
#  COMPONENT 2 — LOAD BALANCER
# ============================================================

class LoadBalancer:
    def __init__(self, routing_probs):
        self.routing_probs = routing_probs

    def route(self, j):
        return np.random.choice(len(self.routing_probs[j]), p=self.routing_probs[j])


# ============================================================
#  COMPONENT 3 — SERVER
# ============================================================

class Server:
    def __init__(self, sid, cpu_cap, access_cap):
        self.sid = sid
        self.cpu_cap = cpu_cap
        self.access_cap = access_cap
        self.active_flows = []

    def current_load(self):
        return len(self.active_flows)

    def current_bw(self):
        return sum(flow["bitrate"] for flow in self.active_flows)

    def add_flow(self, flow):
        self.active_flows.append(flow)

    def remove_finished(self, t):
        self.active_flows = [f for f in self.active_flows if f["t_end"] > t]


# ============================================================
#  COMPONENT 4 — ADMISSION CONTROL
# ============================================================

class AdmissionControl:
    def __init__(self, policy):
        self.policy = policy

    def decide(self, server, flow, apps):
        return self.policy(server, flow, apps)


# Policy with detailed reasons
def policy_with_reasons(server, flow, apps):
    if server.current_load() >= server.cpu_cap:
        return False, "CPU FULL"
    if server.current_bw() + flow["bitrate"] > server.access_cap:
        return False, "BW FULL"
    return True, "ACCEPTED"


# ============================================================
#  COMPONENT 5 — APPLICATION
# ============================================================

class Application:
    def __init__(self, app_id, interest, util_func):
        self.app_id = app_id
        self.interest = interest
        self.util_func = util_func

    def utility(self, w):
        return self.util_func(w)


def simple_utility(w):
    return np.exp(-0.1 * w)


# ============================================================
#  EVENT-DRIVEN SIMULATOR WITH LOGS + CSV
# ============================================================

class Simulator:
    def __init__(self, traffic, balancer, servers, apps, admission):
        self.traffic = traffic
        self.balancer = balancer
        self.servers = servers
        self.apps = apps
        self.admission = admission
        self.t = 0

        # Stats
        self.accepted = 0
        self.rejected = 0
        self.rejected_reasons = {"CPU FULL": 0, "BW FULL": 0}
        self.time_log = []
        self.accept_rate_log = []
        self.server_load_log = [[] for _ in servers]
        self.server_bw_log = [[] for _ in servers]

        # CSV storage
        self.flow_records = []

    def log_state(self):
        self.time_log.append(self.t)
        total = self.accepted + self.rejected
        rate = self.accepted / total if total > 0 else 0
        self.accept_rate_log.append(rate)
        for i, s in enumerate(self.servers):
            self.server_load_log[i].append(s.current_load())
            self.server_bw_log[i].append(s.current_bw())

    def save_csv(self):
        os.makedirs("results", exist_ok=True)
        with open("results/simulation_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "class", "server", "bitrate", "decision", "reason"])
            for row in self.flow_records:
                writer.writerow(row)

    def run(self, T_end):
        while self.t < T_end:
            dt, j = self.traffic.next_arrival()
            self.t += dt

            # Remove expired flows
            for s in self.servers:
                s.remove_finished(self.t)

            # Routing
            i = self.balancer.route(j)
            server = self.servers[i]

            flow = {
                "class": j,
                "bitrate": self.traffic.bitrate(),
                "t_end": self.t + self.traffic.flow_duration(j)
            }

            # Admission
            ok, reason = self.admission.policy(server, flow, self.apps)

            if ok:
                server.add_flow(flow)
                self.accepted += 1
                print(f"[{self.t:.2f}] ACCEPTED c{j} -> S{i} | BW={flow['bitrate']}")
                decision = "ACCEPTED"
            else:
                self.rejected += 1
                self.rejected_reasons[reason] += 1
                print(f"[{self.t:.2f}] REJECTED c{j} -> S{i} | REASON: {reason}")
                decision = "REJECTED"

            # Save CSV entry
            self.flow_records.append([self.t, j, i, flow["bitrate"], decision, reason])

            # Log system state
            self.log_state()

        self.save_csv()

        total = self.accepted + self.rejected
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accept_rate": self.accepted / total,
            "reject_breakdown": self.rejected_reasons
        }


# ============================================================
#  GRAPHICS
# ============================================================

def plot_results(sim, servers):
    t = sim.time_log

    # -------------------- CPU LOAD --------------------
    plt.figure(figsize=(12,6))
    for i in range(len(servers)):
        plt.plot(t, sim.server_load_log[i], label=f"Server {i}")
    plt.title("CPU Load Over Time")
    plt.xlabel("Time")
    plt.ylabel("Active Flows")
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------- BANDWIDTH --------------------
    plt.figure(figsize=(12,6))
    for i in range(len(servers)):
        plt.plot(t, sim.server_bw_log[i], label=f"Server {i}")
    plt.title("Bandwidth Usage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Bandwidth")
    plt.legend()
    plt.grid()
    plt.show()

    # -------------------- ACCEPTANCE RATE --------------------
    plt.figure(figsize=(12,6))
    plt.plot(t, sim.accept_rate_log, color='green')
    plt.title("Acceptance Rate Over Time")
    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.grid()
    plt.show()

    # -------------------- HISTOGRAM CPU --------------------
    plt.figure(figsize=(12,6))
    for i in range(len(servers)):
        plt.hist(sim.server_load_log[i], alpha=0.5, bins=10, label=f"Server {i}")
    plt.title("Histogram of CPU Load")
    plt.legend()
    plt.show()

    # -------------------- HISTOGRAM BW --------------------
    plt.figure(figsize=(12,6))
    for i in range(len(servers)):
        plt.hist(sim.server_bw_log[i], alpha=0.5, bins=10, label=f"Server {i}")
    plt.title("Histogram of Bandwidth")
    plt.legend()
    plt.show()


# ============================================================
#  MAIN PROGRAM WITH REALISTIC REJECTIONS
# ============================================================

if __name__ == "__main__":

    M = 3
    S = 3

    # Strong load (to generate rejections)
    lambdas = [3.0, 2.0, 3.5]
    mu = [0.2, 0.25, 0.20]     # flows stay longer
    cpu_caps = [2, 3, 2]       # small CPU
    access_caps = [6, 7, 5]    # small bandwidth

    traffic = TrafficGenerator(M, lambdas, mu,
                               bitrate_dist=lambda: np.random.randint(2, 6))

    routing_probs = [
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.3, 0.3, 0.4],
    ]

    balancer = LoadBalancer(routing_probs)
    servers = [Server(i, cpu_caps[i], access_caps[i]) for i in range(S)]

    apps = [
        Application(0, [1, 1, 0], simple_utility),
        Application(1, [0, 1, 1], simple_utility)
    ]

    admission = AdmissionControl(policy_with_reasons)

    sim = Simulator(traffic, balancer, servers, apps, admission)
    results = sim.run(200)

    print("\n===== FINAL RESULTS =====")
    print(results)
    print("\nBreakdown of rejections:", results["reject_breakdown"])

    plot_results(sim, servers)
