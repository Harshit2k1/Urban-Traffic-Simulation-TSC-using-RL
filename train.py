from __future__ import absolute_import, print_function

import os, sys, pathlib
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- SUMO setup ----------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME to your SUMO install dir")

from sumolib import checkBinary
import traci


# ---------- utility helpers ----------
def get_vehicle_numbers(lanes):
    """Number of vehicles per lane that are >10 m from the stopline."""
    return {
        l: sum(1 for v in traci.lane.getLastStepVehicleIDs(l)
               if traci.vehicle.getLanePosition(v) > 10)
        for l in lanes
    }

def get_waiting_time(lanes):
    """Total waiting time of all vehicles on a lane set (SUMO API)."""
    return sum(traci.lane.getWaitingTime(l) for l in lanes)

def phaseDuration(junction, phase_time, phase_state):
    """Set the traffic light to *phase_state* for *phase_time* seconds."""
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


# ---------- deep‑Q network ----------
class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims,  fc2_dims)
        self.linear3 = nn.Linear(fc2_dims,  n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss     = nn.MSELoss()
        self.device   = torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


# ---------- replay‑buffer agent ----------
class Agent:
    def __init__(self, gamma, epsilon, lr,
                 input_dims, fc1_dims, fc2_dims,
                 batch_size, n_actions, junctions,
                 max_memory_size=100000,
                 epsilon_dec=5e-4, epsilon_end=0.05):

        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.Q_eval      = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)

        self.action_space = list(range(n_actions))
        self.junctions    = junctions
        self.memory = {
            j: {
                "state"     : np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "new_state" : np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "reward"    : np.zeros(max_memory_size, dtype=np.float32),
                "action"    : np.zeros(max_memory_size, dtype=np.int32),
                "done"      : np.zeros(max_memory_size, dtype=bool),
                "counter"   : 0
            } for j in junctions
        }

    # ---- buffer helpers ----
    def store_transition(self, state, state_, action, reward, done, junction):
        buf = self.memory[junction]
        idx = buf["counter"] % buf["state"].shape[0]
        buf["state"][idx]     = state
        buf["new_state"][idx] = state_
        buf["reward"][idx]    = reward
        buf["action"][idx]    = action
        buf["done"][idx]      = done
        buf["counter"] += 1

    # ---- policy ----
    def choose_action(self, obs, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        state = torch.tensor([obs], dtype=torch.float32,
                             device=self.Q_eval.device)
        actions = self.Q_eval(state)
        return torch.argmax(actions).item()

    # ---- TD learning ----
    def learn(self, junction):
        buf = self.memory[junction]
        cnt = buf["counter"]
        if cnt == 0:
            return

        # simple “all‑at‑once” replay
        idxs = np.arange(cnt, dtype=np.int64)
        s_batch  = torch.tensor(buf["state"][idxs],     device=self.Q_eval.device)
        ns_batch = torch.tensor(buf["new_state"][idxs], device=self.Q_eval.device)
        r_batch  = torch.tensor(buf["reward"][idxs],    device=self.Q_eval.device)
        d_batch  = torch.tensor(buf["done"][idxs],      device=self.Q_eval.device)
        a_batch  = buf["action"][idxs]

        q_eval  = self.Q_eval(s_batch)[idxs, a_batch]
        q_next  = self.Q_eval(ns_batch)
        q_next[d_batch] = 0.0
        q_target = r_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        # ε‑decay
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)


# ---------- full training run ----------
def run(
    controlled_junctions,
    model_name: str,
    epochs: int,
    steps: int,
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 0.0
):

    agent = Agent(
        gamma=0.99, epsilon=0.1, lr=0.01,
        input_dims=4, fc1_dims=256, fc2_dims=256,
        batch_size=512, n_actions=4,
        junctions=list(range(len(controlled_junctions)))
    )

    # ---- tracking arrays ----
    action_counts = np.zeros((epochs, agent.Q_eval.linear3.out_features),
                             dtype=int)

    best_wait     = float("inf")
    best_epoch    = 0
    no_improve    = 0
    actual_epochs = epochs   # may shorten if early‑stop

    # ---- training loop ----
    for e in range(epochs):
        actual_epochs = e + 1

        traci.start([
            checkBinary("sumo"),
            "-c", "configuration.sumocfg",
            "--tripinfo-output", f"tripinfo_epoch{e}.xml",
            "--no-warnings", "true"
        ])

        total_wait = 0
        step       = 0
        prev_veh   = {i: [0]*4 for i in range(len(controlled_junctions))}
        prev_act   = {i: 0     for i in range(len(controlled_junctions))}
        light_t    = {j: 0     for j in controlled_junctions}

        while step < steps:
            traci.simulationStep()

            for jn, junction in enumerate(controlled_junctions):
                lanes = traci.trafficlight.getControlledLanes(junction)
                w = get_waiting_time(lanes)
                total_wait += w

                if light_t[junction] == 0:
                    veh_counts = list(get_vehicle_numbers(lanes).values())
                    reward = -w

                    agent.store_transition(prev_veh[jn], veh_counts,
                                           prev_act[jn], reward,
                                           False, jn)

                    action = agent.choose_action(veh_counts)
                    action_counts[e, action] += 1
                    prev_act[jn] = action
                    prev_veh[jn] = veh_counts

                    from_phase, to_phase = SELECT_PHASES[action]
                    phaseDuration(junction, 6,  from_phase)
                    phaseDuration(junction, 15, to_phase)
                    light_t[junction] = 15

                    agent.learn(jn)
                else:
                    light_t[junction] -= 1
            step += 1

        traci.close()

        print(f"[Epoch {e+1}/{epochs}] Total wait time: {total_wait:.1f}")
        if total_wait < best_wait - early_stopping_delta:
            best_wait  = total_wait
            best_epoch = e + 1
            no_improve = 0
            pathlib.Path("models").mkdir(exist_ok=True)
            torch.save(agent.Q_eval.state_dict(),
                       f"models/{model_name}.bin")
            print(f"  ↳ New best at epoch {best_epoch} "
                  f"(wait={total_wait:.1f}), model saved.")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  ↳ No improvement in {early_stopping_patience} "
                      "epochs; early stopping.")
                break

    # ------------- plotting & saving metrics -------------
    xs = np.arange(1, actual_epochs+1)

    pathlib.Path("plots").mkdir(exist_ok=True)

    # ---- 1. RAW COUNTS ----
    plt.figure(figsize=(8, 5))
    for a in range(action_counts.shape[1]):
        plt.plot(xs, action_counts[:actual_epochs, a], label=f"Action {a}")
    plt.axvline(best_epoch, linestyle=':', label='Best Epoch')
    plt.axvline(actual_epochs, linestyle='--', label='Early Stop')
    plt.xlabel("Epoch")
    plt.ylabel("Times Selected")
    plt.title("Action Selection Counts per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/action_counts.png")
    plt.close()
    print("Saved raw count chart to plots/action_counts.png")

    # ---- 2. PROPORTIONS ----
    totals = action_counts[:actual_epochs].sum(axis=1, keepdims=True)
    # Avoid divide‑by‑zero where no action was chosen
    action_props = np.divide(action_counts[:actual_epochs], totals,
                             where=totals != 0)

    plt.figure(figsize=(8, 5))
    for a in range(action_props.shape[1]):
        plt.plot(xs, action_props[:, a], label=f"Action {a}")
    plt.axvline(best_epoch, linestyle=':', label='Best Epoch')
    plt.axvline(actual_epochs, linestyle='--', label='Early Stop')
    plt.xlabel("Epoch")
    plt.ylabel("Proportion Selected")
    plt.title("Action Selection *Proportions* per Epoch")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("plots/action_proportions.png")
    plt.close()
    print("Saved proportion chart to plots/action_proportions.png")

    # ---- 3. Numpy dumps for offline analysis ----
    np.save("action_counts.npy",     action_counts[:actual_epochs])
    np.save("action_proportions.npy", action_props)
    print("Saved arrays to action_counts.npy, action_proportions.npy")


# ---------- main ----------
if __name__ == "__main__":
    # phase definitions for each of the 4 actions
    SELECT_PHASES = [
        ("rrrrGGggrrrrGGgg", "GGGGrrrrGGGGrrrr"),
        ("rrrryyyyrrrryyyy", "rrrrGGGGrrrrGGGG"),
        ("GGggrrrrGGggrrrr", "GGGGrrrrGGGGrrrr"),
        ("yyyyrrrryyyyrrrr", "rrrrGGGGrrrrGGGG"),
    ]

    junctions_to_monitor = [
        '392163460', '392170608', '392180247', '249969885'
    ]

    run(
        controlled_junctions=junctions_to_monitor,
        model_name="city_tsc1",
        epochs=150,
        steps=1000,
        early_stopping_patience=10,
        early_stopping_delta=0.0
    )
