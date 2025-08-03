#!/usr/bin/env python3
from __future__ import absolute_import, print_function

import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import optuna

# SUMO setup
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME to your SUMO install dir")

from sumolib import checkBinary
import traci

# ---------------------------------------------------------------------------
# Helpers: Traffic metrics & phase control
# ---------------------------------------------------------------------------
def get_vehicle_numbers(lanes):
    return {
        l: sum(
            1
            for v in traci.lane.getLastStepVehicleIDs(l)
            if traci.vehicle.getLanePosition(v) > 10
        )
        for l in lanes
    }

def get_waiting_time(lanes):
    return sum(traci.lane.getWaitingTime(l) for l in lanes)

def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

# ---------------------------------------------------------------------------
# Q‐Network & DDQN Agent
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, lr, input_dims, hidden_dims, n_actions):
        super().__init__()
        layers = []
        in_dim = input_dims
        # build a dynamic MLP based on hidden_dims list
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss      = nn.MSELoss()
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        epsilon_dec,
        epsilon_end,
        lr,
        input_dims,
        hidden_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=100000
    ):
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size  = batch_size

        # dynamic Q‐network
        self.Q_eval       = Model(lr, input_dims, hidden_dims, n_actions)
        self.action_space = list(range(n_actions))
        self.junctions    = junctions

        # Per‐junction replay buffers
        self.memory = {
            j: {
                "state"     : np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "new_state" : np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "reward"    : np.zeros(max_memory_size, dtype=np.float32),
                "action"    : np.zeros(max_memory_size, dtype=np.int32),
                "done"      : np.zeros(max_memory_size, dtype=bool),
                "counter"   : 0
            }
            for j in junctions
        }

    def store_transition(self, state, state_, action, reward, done, junction):
        buf = self.memory[junction]
        idx = buf["counter"] % buf["state"].shape[0]
        buf["state"][idx]     = state
        buf["new_state"][idx] = state_
        buf["reward"][idx]    = reward
        buf["action"][idx]    = action
        buf["done"][idx]      = done
        buf["counter"]       += 1

    def choose_action(self, obs, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        state_tensor = torch.tensor([obs], dtype=torch.float32, device=self.Q_eval.device)
        actions = self.Q_eval(state_tensor)
        return torch.argmax(actions).item()

    def learn(self, junction):
        buf = self.memory[junction]
        cnt = buf["counter"]
        if cnt == 0:
            return

        # sample all stored transitions
        idxs = np.arange(min(cnt, buf["state"].shape[0]), dtype=np.int64)
        s_batch  = torch.tensor(buf["state"][idxs],     device=self.Q_eval.device)
        ns_batch = torch.tensor(buf["new_state"][idxs], device=self.Q_eval.device)
        r_batch  = torch.tensor(buf["reward"][idxs],    device=self.Q_eval.device)
        d_batch  = torch.tensor(buf["done"][idxs],      device=self.Q_eval.device)
        a_batch  = buf["action"][idxs]

        q_eval = self.Q_eval(s_batch)[idxs, a_batch]
        q_next = self.Q_eval(ns_batch)
        q_next[d_batch] = 0.0
        q_target = r_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

# ---------------------------------------------------------------------------
# Single‐episode runner
# ---------------------------------------------------------------------------
def run_episode(agent, controlled_junctions, steps, sumo_cfg="configuration.sumocfg"):
    sumo_cmd = [
        checkBinary("sumo"),
        "-c", sumo_cfg,
        "--no-warnings", "true",
        "--no-step-log", "true"
    ]
    traci.start(sumo_cmd)

    total_wait = 0
    step = 0
    prev_veh = {i: [0]*4 for i in range(len(controlled_junctions))}
    prev_act = {i: 0     for i in range(len(controlled_junctions))}
    light_t  = {j: 0     for j in controlled_junctions}

    while step < steps:
        traci.simulationStep()
        for jn, junction in enumerate(controlled_junctions):
            lanes = traci.trafficlight.getControlledLanes(junction)
            w = get_waiting_time(lanes)
            total_wait += w

            if light_t[junction] == 0:
                veh_counts = list(get_vehicle_numbers(lanes).values())
                reward = -w

                agent.store_transition(
                    prev_veh[jn],
                    veh_counts,
                    prev_act[jn],
                    reward,
                    False,
                    jn
                )

                action = agent.choose_action(veh_counts)
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
    return total_wait

# ---------------------------------------------------------------------------
# Optuna objective with dynamic architecture
# ---------------------------------------------------------------------------
def objective(trial):
    # hyperparameter search space
    gamma       = trial.suggest_float("gamma",        0.90,  0.999,   step=0.001)
    epsilon     = trial.suggest_float("epsilon",      0.01,  0.2)
    epsilon_end = trial.suggest_float("epsilon_end",  0.01,  0.10)
    epsilon_dec = trial.suggest_loguniform("epsilon_dec", 1e-5, 1e-2)
    lr          = trial.suggest_loguniform("lr",       1e-5,  1e-1)

    # continuous vs categorical layer sizes & number of layers
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f"hidden_dim_{i+1}", 32, 1024, log=True))

    batch_size  = trial.suggest_int("batch_size", 128, 1024, log=True)

    # build agent with dynamic MLP
    agent = Agent(
        gamma=gamma,
        epsilon=epsilon,
        epsilon_dec=epsilon_dec,
        epsilon_end=epsilon_end,
        lr=lr,
        input_dims=4,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        n_actions=4,
        junctions=list(range(len(junctions_to_monitor)))
    )

    # run 3 episodes and sum total wait
    waits = [run_episode(agent, junctions_to_monitor, steps=1000) for _ in range(3)]
    total_wait = sum(waits)
    return total_wait  # minimize total wait

# ---------------------------------------------------------------------------
# Main: define phases, junctions, and start Optuna study with Hyperband pruner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SELECT_PHASES = [
        ("rrrrGGggrrrrGGgg", "GGGGrrrrGGGGrrrr"),
        ("rrrryyyyrrrryyyy", "rrrrGGGGrrrrGGGG"),
        ("GGggrrrrGGggrrrr", "GGGGrrrrGGGGrrrr"),
        ("yyyyrrrryyyyrrrr", "rrrrGGGGrrrrGGGG"),
    ]
    junctions_to_monitor = ['392163460','392170608','392180247','249969885']

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=720, timeout=3600)

    print("Best hyperparameters:", study.best_params)
    print("Lowest total wait:     ", study.best_value)
