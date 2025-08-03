#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
traffic_simulation.py
Runs four Traffic Signal Control algorithms, records both run-level 
statisticsand per-step metrics, then saves every raw array needed for
down-stream visualisation to comparison_plots/per_step_results.npz
"""

from __future__ import absolute_import, print_function

import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME environment variable not defined")

from sumolib import checkBinary
import traci


# ---------------------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_vehicle_numbers(lanes):
    return {
        l: sum(
            1
            for k in traci.lane.getLastStepVehicleIDs(l)
            if traci.vehicle.getLanePosition(k) > 10
        )
        for l in lanes
    }


def phaseDuration(junction, phase_time, phase_state):
    try:
        traci.trafficlight.setRedYellowGreenState(junction, phase_state)
        traci.trafficlight.setPhaseDuration(junction, phase_time)
    except Exception as e:
        print(f"Error: {e}")


def get_all_vehicle_wait_times():
    veh_wait = {}
    for veh_id in traci.vehicle.getIDList():
        try:
            veh_wait[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        except Exception:
            continue
    return veh_wait


# ---------------------------------------------------------------------
#  MODEL & AGENT  
# ---------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )

    def choose_action(self, observation, evaluate=False):
        try:
            state = torch.tensor([observation], dtype=torch.float).to(
                self.Q_eval.device
            )
            if evaluate or np.random.random() > self.epsilon:
                actions = self.Q_eval.forward(state)
                action = torch.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
            return action
        except Exception as e:
            print(f"Error occurred while choosing action: {e}")
            return np.random.choice(self.action_space)

    def load(self, model_name):
        try:
            self.Q_eval.load_state_dict(torch.load(f"models/{model_name}.bin"))
            self.Q_eval.eval()
            print(f"Model {model_name} loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# ---------------------------------------------------------------------
#  MAX PRESSURE 
# ---------------------------------------------------------------------
def get_downstream_lanes(lane):
    try:
        links = traci.lane.getLinks(lane)
        return [link[0] for link in links]
    except:
        return []


def calculate_pressure(lane):
    try:
        queue_length = traci.lane.getLastStepVehicleNumber(lane)
        downstream_lanes = get_downstream_lanes(lane)
        downstream_queue = sum(
            traci.lane.getLastStepVehicleNumber(dl) for dl in downstream_lanes if dl
        )
        return max(0, queue_length - downstream_queue)
    except:
        return 0


def run_sumo_with_max_pressure(junctions_to_monitor, steps=1000, num_runs=10, seed=42):
    np.random.seed(seed)
    os.makedirs("max_pressure_results", exist_ok=True)
    MIN_GREEN = 8
    AMBER_TIME = 4

    avg_wait_times, throughputs, total_wait_times = [], [], []
    per_step_avg_runs, per_step_total_runs = [], []

    for run in range(num_runs):
        print(f"\nMax Pressure Run {run+1}/{num_runs}")
        run_seed = seed + run
        traci.start(
            [
                checkBinary("sumo"),
                "-c",
                "configuration.sumocfg",
                "--tripinfo-output",
                f"max_pressure_results/tripinfo_mp_run{run}.xml",
                "--seed",
                f"{run_seed}",
            ]
        )

        step = 0
        timer = {j: 0 for j in junctions_to_monitor}
        latest_wait_times = {}

        per_step_avg_series = []
        per_step_total_series = []

        while step < steps:
            traci.simulationStep()
            for j in junctions_to_monitor:
                lanes = traci.trafficlight.getControlledLanes(j)
                if timer[j] == 0:
                    queues = list(get_vehicle_numbers(lanes).values())[:4]
                    pressures = [queues[i] - queues[(i + 2) % 4] for i in range(4)]
                    ph = int(np.argmax(pressures))
                    phaseDuration(
                        j,
                        MIN_GREEN,
                        [
                            "rrrrGGggrrrrGGgg",
                            "GGGGrrrrGGGGrrrr",
                            "GGggrrrrGGggrrrr",
                            "rrrrGGGGrrrrGGGG",
                        ][ph],
                    )
                    phaseDuration(
                        j,
                        AMBER_TIME,
                        [
                            "GGGGrrrrGGGGrrrr",
                            "rrrrGGGGrrrrGGGG",
                            "GGGGrrrrGGGGrrrr",
                            "rrrrGGGGrrrrGGGG",
                        ][ph],
                    )
                    timer[j] = MIN_GREEN + AMBER_TIME
                else:
                    timer[j] -= 1

            current_waits = list(get_all_vehicle_wait_times().values())
            per_step_avg_series.append(
                np.mean(current_waits) if current_waits else 0.0
            )
            per_step_total_series.append(np.sum(current_waits))

            for veh_id, wait_time in get_all_vehicle_wait_times().items():
                latest_wait_times[veh_id] = wait_time

            step += 1

        # run‑level statistics
        all_wait_times = list(latest_wait_times.values())
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0
        total_wait_time = np.sum(all_wait_times)
        throughput = len(latest_wait_times)

        avg_wait_times.append(avg_wait_time)
        throughputs.append(throughput)
        total_wait_times.append(total_wait_time)

        per_step_avg_runs.append(per_step_avg_series)
        per_step_total_runs.append(per_step_total_series)

        print(
            f"Run {run+1} Results:\n"
            f"  Total wait time: {total_wait_time}\n"
            f"  Average wait time per vehicle: {avg_wait_time:.2f}\n"
            f"  Total unique vehicle throughput: {throughput}"
        )

        traci.close()

    # aggregate statistics 
    confidence_level = 0.95
    avg_wait_time_mean = np.mean(avg_wait_times)
    avg_wait_time_ci = stats.t.interval(
        confidence_level,
        len(avg_wait_times) - 1,
        loc=avg_wait_time_mean,
        scale=stats.sem(avg_wait_times),
    )
    throughput_mean = np.mean(throughputs)
    throughput_ci = stats.t.interval(
        confidence_level,
        len(throughputs) - 1,
        loc=throughput_mean,
        scale=stats.sem(throughputs),
    )
    total_wait_time_mean = np.mean(total_wait_times)
    total_wait_time_ci = stats.t.interval(
        confidence_level,
        len(total_wait_times) - 1,
        loc=total_wait_time_mean,
        scale=stats.sem(total_wait_times),
    )

    print("\n===== MAX PRESSURE RESULTS =====")
    print(
        f"Average wait time: {avg_wait_time_mean:.2f} "
        f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]"
    )
    print(
        f"Total unique vehicle throughput: {throughput_mean:.2f} "
        f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]"
    )
    print(
        f"Total wait time: {total_wait_time_mean:.2f} "
        f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]"
    )

    with open("max_pressure_results/max_pressure_results.txt", "w") as f:
        f.write("===== MAX PRESSURE RESULTS =====\n")
        f.write(f"Number of test runs: {num_runs}\n")
        f.write(
            f"Average wait time: {avg_wait_time_mean:.2f} "
            f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total unique vehicle throughput: {throughput_mean:.2f} "
            f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total wait time: {total_wait_time_mean:.2f} "
            f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]\n"
        )

    return {
        "avg_wait_time": {
            "mean": avg_wait_time_mean,
            "ci_lower": avg_wait_time_ci[0],
            "ci_upper": avg_wait_time_ci[1],
        },
        "throughput": {
            "mean": throughput_mean,
            "ci_lower": throughput_ci[0],
            "ci_upper": throughput_ci[1],
        },
        "total_wait_time": {
            "mean": total_wait_time_mean,
            "ci_lower": total_wait_time_ci[0],
            "ci_upper": total_wait_time_ci[1],
        },
        "raw_data": {
            "avg_wait_times": avg_wait_times,
            "throughputs": throughputs,
            "total_wait_times": total_wait_times,
        },
        "per_step": {
            "avg": per_step_avg_runs,  # shape (runs, steps+1)
            "total": per_step_total_runs,
        },
    }


# ---------------------------------------------------------------------
#  WEBSTER'S METHOD 
# ---------------------------------------------------------------------
def calculate_saturation_flow(lane):
    try:
        base_flow = 1900
        width_factor = min(max(traci.lane.getWidth(lane) / 3.5, 0.9), 1.1)
        speed_factor = min(max(traci.lane.getMaxSpeed(lane) / 15, 0.9), 1.1)
        return base_flow * width_factor * speed_factor
    except:
        return 1800


def calculate_flow_ratio(lane):
    try:
        vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
        mean_speed = traci.lane.getLastStepMeanSpeed(lane)
        speed_factor = 2.0 if mean_speed < 2.0 else 1.0
        adjusted_count = vehicle_count * speed_factor
        flow_rate = adjusted_count * 3600 / 60
        return flow_rate / calculate_saturation_flow(lane)
    except:
        return 0


def run_sumo_with_webster(junctions_to_monitor, steps=1000, num_runs=10, seed=42):
    np.random.seed(seed)
    os.makedirs("webster_results", exist_ok=True)
    MIN_GREEN = 5
    AMBER_TIME = 3
    WEBSTER_CYCLE = 60

    avg_wait_times, throughputs, total_wait_times = [], [], []
    per_step_avg_runs, per_step_total_runs = [], []

    for run in range(num_runs):
        print(f"\nWebster Run {run+1}/{num_runs}")
        run_seed = seed + run
        traci.start(
            [
                checkBinary("sumo"),
                "-c",
                "configuration.sumocfg",
                "--tripinfo-output",
                f"webster_results/tripinfo_webster_run{run}.xml",
                "--seed",
                f"{run_seed}",
            ]
        )

        step = 0
        timer = {j: 0 for j in junctions_to_monitor}
        latest_wait_times = {}

        per_step_avg_series = []
        per_step_total_series = []

        while step < steps:
            traci.simulationStep()
            for j in junctions_to_monitor:
                lanes = traci.trafficlight.getControlledLanes(j)
                if timer[j] == 0:
                    queues = list(get_vehicle_numbers(lanes).values())[:4]
                    total_q = sum(queues)
                    greens = [
                        max(MIN_GREEN, int((q / total_q) * WEBSTER_CYCLE))
                        if total_q > 0
                        else MIN_GREEN
                        for q in queues
                    ]
                    ph = int(np.argmax(greens))
                    phaseDuration(
                        j,
                        greens[ph],
                        [
                            "rrrrGGggrrrrGGgg",
                            "GGGGrrrrGGGGrrrr",
                            "GGggrrrrGGggrrrr",
                            "rrrrGGGGrrrrGGGG",
                        ][ph],
                    )
                    phaseDuration(
                        j,
                        AMBER_TIME,
                        [
                            "GGGGrrrrGGGGrrrr",
                            "rrrrGGGGrrrrGGGG",
                            "GGGGrrrrGGGGrrrr",
                            "rrrrGGGGrrrrGGGG",
                        ][ph],
                    )
                    timer[j] = greens[ph] + AMBER_TIME
                else:
                    timer[j] -= 1

            current_waits = list(get_all_vehicle_wait_times().values())
            per_step_avg_series.append(
                np.mean(current_waits) if current_waits else 0.0
            )
            per_step_total_series.append(np.sum(current_waits))

            for veh_id, wait_time in get_all_vehicle_wait_times().items():
                latest_wait_times[veh_id] = wait_time

            step += 1

        all_wait_times = list(latest_wait_times.values())
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0
        total_wait_time = np.sum(all_wait_times)
        throughput = len(latest_wait_times)

        avg_wait_times.append(avg_wait_time)
        throughputs.append(throughput)
        total_wait_times.append(total_wait_time)

        per_step_avg_runs.append(per_step_avg_series)
        per_step_total_runs.append(per_step_total_series)

        print(
            f"Run {run+1} Results:\n"
            f"  Total wait time: {total_wait_time}\n"
            f"  Average wait time per vehicle: {avg_wait_time:.2f}\n"
            f"  Total unique vehicle throughput: {throughput}"
        )

        traci.close()

    confidence_level = 0.95
    avg_wait_time_mean = np.mean(avg_wait_times)
    avg_wait_time_ci = stats.t.interval(
        confidence_level,
        len(avg_wait_times) - 1,
        loc=avg_wait_time_mean,
        scale=stats.sem(avg_wait_times),
    )
    throughput_mean = np.mean(throughputs)
    throughput_ci = stats.t.interval(
        confidence_level,
        len(throughputs) - 1,
        loc=throughput_mean,
        scale=stats.sem(throughputs),
    )
    total_wait_time_mean = np.mean(total_wait_times)
    total_wait_time_ci = stats.t.interval(
        confidence_level,
        len(total_wait_times) - 1,
        loc=total_wait_time_mean,
        scale=stats.sem(total_wait_times),
    )

    print("\n===== WEBSTER'S METHOD RESULTS =====")
    print(
        f"Average wait time: {avg_wait_time_mean:.2f} "
        f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]"
    )
    print(
        f"Total unique vehicle throughput: {throughput_mean:.2f} "
        f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]"
    )
    print(
        f"Total wait time: {total_wait_time_mean:.2f} "
        f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]"
    )

    with open("webster_results/webster_results.txt", "w") as f:
        f.write("===== WEBSTER'S METHOD RESULTS =====\n")
        f.write(f"Number of test runs: {num_runs}\n")
        f.write(
            f"Average wait time: {avg_wait_time_mean:.2f} "
            f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total unique vehicle throughput: {throughput_mean:.2f} "
            f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total wait time: {total_wait_time_mean:.2f} "
            f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]\n"
        )

    return {
        "avg_wait_time": {
            "mean": avg_wait_time_mean,
            "ci_lower": avg_wait_time_ci[0],
            "ci_upper": avg_wait_time_ci[1],
        },
        "throughput": {
            "mean": throughput_mean,
            "ci_lower": throughput_ci[0],
            "ci_upper": throughput_ci[1],
        },
        "total_wait_time": {
            "mean": total_wait_time_mean,
            "ci_lower": total_wait_time_ci[0],
            "ci_upper": total_wait_time_ci[1],
        },
        "raw_data": {
            "avg_wait_times": avg_wait_times,
            "throughputs": throughputs,
            "total_wait_times": total_wait_times,
        },
        "per_step": {"avg": per_step_avg_runs, "total": per_step_total_runs},
    }


# ---------------------------------------------------------------------
#  RL MODEL TEST + BASELINE
# ---------------------------------------------------------------------
def test_rl_model(
    controlled_junctions, model_name, steps=1000, num_runs=10, seed=42
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs("test_results", exist_ok=True)

    junction_numbers = list(range(len(controlled_junctions)))

    brain = Agent(
        gamma=0.99,
        epsilon=0,
        lr=0.01,
        input_dims=4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=4,
        junctions=junction_numbers,
    )

    if not brain.load(model_name):
        print(f"Failed to load model {model_name}. Exiting test.")
        return None

    print(f"Testing model {model_name} on device: {brain.Q_eval.device}")

    select_lane = [
        ["rrrrGGggrrrrGGgg", "GGGGrrrrGGGGrrrr"],
        ["rrrryyyyrrrryyyy", "rrrrGGGGrrrrGGGG"],
        ["GGggrrrrGGggrrrr", "GGGGrrrrGGGGrrrr"],
        ["yyyyrrrryyyyrrrr", "rrrrGGGGrrrrGGGG"],
    ]

    avg_wait_times, throughputs, total_wait_times = [], [], []
    per_step_avg_runs, per_step_total_runs = [], []

    for run in range(num_runs):
        print(f"\nRL Test Run {run+1}/{num_runs}")
        run_seed = seed + run
        traci.start(
            [
                checkBinary("sumo"),
                "-c",
                "configuration.sumocfg",
                "--tripinfo-output",
                f"test_results/tripinfo_test_run{run}.xml",
                "--seed",
                f"{run_seed}",
            ]
        )

        step = 0
        min_duration = 5

        traffic_lights_time = {}
        prev_vehicles_per_lane = {}
        all_lanes = []
        action_counts = np.zeros(4)
        latest_wait_times = {}

        for junction_number, junction in enumerate(controlled_junctions):
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 4
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        per_step_avg_series = []
        per_step_total_series = []

        while step < steps:
            traci.simulationStep()

            for junction_number, junction in enumerate(controlled_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    state_ = list(vehicles_per_lane.values())
                    prev_vehicles_per_lane[junction_number] = state_

                    lane = brain.choose_action(state_, evaluate=True)

                    phaseDuration(junction, 6, select_lane[lane][0])
                    phaseDuration(junction, min_duration + 10, select_lane[lane][1])
                    traffic_lights_time[junction] = min_duration + 10

                    action_counts[lane] += 1
                else:
                    traffic_lights_time[junction] -= 1

            # --- NEW: per‑step metrics ------------------------------
            current_waits = list(get_all_vehicle_wait_times().values())
            per_step_avg_series.append(
                np.mean(current_waits) if current_waits else 0.0
            )
            per_step_total_series.append(np.sum(current_waits))
            # ---------------------------------------------------------

            for veh_id, wait_time in get_all_vehicle_wait_times().items():
                latest_wait_times[veh_id] = wait_time

            step += 1

        all_wait_times = list(latest_wait_times.values())
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0
        total_wait_time = np.sum(all_wait_times)
        throughput = len(latest_wait_times)

        avg_wait_times.append(avg_wait_time)
        throughputs.append(throughput)
        total_wait_times.append(total_wait_time)

        per_step_avg_runs.append(per_step_avg_series)
        per_step_total_runs.append(per_step_total_series)

        print(
            f"Run {run+1} Results:\n"
            f"  Total wait time: {total_wait_time}\n"
            f"  Average wait time per vehicle: {avg_wait_time:.2f}\n"
            f"  Total unique vehicle throughput: {throughput}"
        )

        traci.close()

    confidence_level = 0.95
    avg_wait_time_mean = np.mean(avg_wait_times)
    avg_wait_time_ci = stats.t.interval(
        confidence_level,
        len(avg_wait_times) - 1,
        loc=avg_wait_time_mean,
        scale=stats.sem(avg_wait_times),
    )
    throughput_mean = np.mean(throughputs)
    throughput_ci = stats.t.interval(
        confidence_level,
        len(throughputs) - 1,
        loc=throughput_mean,
        scale=stats.sem(throughputs),
    )
    total_wait_time_mean = np.mean(total_wait_times)
    total_wait_time_ci = stats.t.interval(
        confidence_level,
        len(total_wait_times) - 1,
        loc=total_wait_time_mean,
        scale=stats.sem(total_wait_times),
    )

    print("\n===== RL MODEL TEST RESULTS =====")
    print(f"Model: {model_name}")
    print(
        f"Average wait time: {avg_wait_time_mean:.2f} "
        f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]"
    )
    print(
        f"Total unique vehicle throughput: {throughput_mean:.2f} "
        f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]"
    )
    print(
        f"Total wait time: {total_wait_time_mean:.2f} "
        f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]"
    )

    with open(f"test_results/rl_results_{model_name}.txt", "w") as f:
        f.write("===== RL MODEL TEST RESULTS =====\n")
        f.write(f"Model: {model_name}\n")
        f.write(
            f"Average wait time: {avg_wait_time_mean:.2f} "
            f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total unique vehicle throughput: {throughput_mean:.2f} "
            f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total wait time: {total_wait_time_mean:.2f} "
            f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]\n"
        )

    return {
        "avg_wait_time": {
            "mean": avg_wait_time_mean,
            "ci_lower": avg_wait_time_ci[0],
            "ci_upper": avg_wait_time_ci[1],
        },
        "throughput": {
            "mean": throughput_mean,
            "ci_lower": throughput_ci[0],
            "ci_upper": throughput_ci[1],
        },
        "total_wait_time": {
            "mean": total_wait_time_mean,
            "ci_lower": total_wait_time_ci[0],
            "ci_upper": total_wait_time_ci[1],
        },
        "raw_data": {
            "avg_wait_times": avg_wait_times,
            "throughputs": throughputs,
            "total_wait_times": total_wait_times,
        },
        "per_step": {"avg": per_step_avg_runs, "total": per_step_total_runs},
    }


def run_sumo_without_rl(junctions_to_monitor, steps=1000, num_runs=10, seed=42):
    np.random.seed(seed)
    os.makedirs("baseline_results", exist_ok=True)

    avg_wait_times, throughputs, total_wait_times = [], [], []
    per_step_avg_runs, per_step_total_runs = [], []

    for run in range(num_runs):
        print(f"\nBaseline Run {run+1}/{num_runs}")

        phase_change_counts = {junction: 0 for junction in junctions_to_monitor}
        previous_phases = {junction: None for junction in junctions_to_monitor}
        latest_wait_times = {}

        run_seed = seed + run
        traci.start(
            [
                checkBinary("sumo"),
                "-c",
                "configuration.sumocfg",
                "--tripinfo-output",
                f"baseline_results/tripinfo_baseline_run{run}.xml",
                "--seed",
                f"{run_seed}",
            ]
        )

        per_step_avg_series = []
        per_step_total_series = []

        for _ in range(steps):
            traci.simulationStep()

            for junction in junctions_to_monitor:
                current_phase = traci.trafficlight.getPhase(junction)
                if (
                    previous_phases[junction] is not None
                    and current_phase != previous_phases[junction]
                ):
                    phase_change_counts[junction] += 1
                previous_phases[junction] = current_phase

            current_waits = list(get_all_vehicle_wait_times().values())
            per_step_avg_series.append(
                np.mean(current_waits) if current_waits else 0.0
            )
            per_step_total_series.append(np.sum(current_waits))
            for veh_id, wait_time in get_all_vehicle_wait_times().items():
                latest_wait_times[veh_id] = wait_time

        all_wait_times = list(latest_wait_times.values())
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0
        total_wait_time = np.sum(all_wait_times)
        throughput = len(latest_wait_times)

        avg_wait_times.append(avg_wait_time)
        throughputs.append(throughput)
        total_wait_times.append(total_wait_time)

        per_step_avg_runs.append(per_step_avg_series)
        per_step_total_runs.append(per_step_total_series)

        print(
            f"Run {run+1} Results:\n"
            f"  Total wait time: {total_wait_time}\n"
            f"  Average wait time per vehicle: {avg_wait_time:.2f}\n"
            f"  Total unique vehicle throughput: {throughput}"
        )

        traci.close()

    confidence_level = 0.95
    avg_wait_time_mean = np.mean(avg_wait_times)
    avg_wait_time_ci = stats.t.interval(
        confidence_level,
        len(avg_wait_times) - 1,
        loc=avg_wait_time_mean,
        scale=stats.sem(avg_wait_times),
    )
    throughput_mean = np.mean(throughputs)
    throughput_ci = stats.t.interval(
        confidence_level,
        len(throughputs) - 1,
        loc=throughput_mean,
        scale=stats.sem(throughputs),
    )
    total_wait_time_mean = np.mean(total_wait_times)
    total_wait_time_ci = stats.t.interval(
        confidence_level,
        len(total_wait_times) - 1,
        loc=total_wait_time_mean,
        scale=stats.sem(total_wait_times),
    )

    print("\n===== BASELINE RESULTS =====")
    print(
        f"Average wait time: {avg_wait_time_mean:.2f} "
        f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]"
    )
    print(
        f"Total unique vehicle throughput: {throughput_mean:.2f} "
        f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]"
    )
    print(
        f"Total wait time: {total_wait_time_mean:.2f} "
        f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]"
    )

    with open("baseline_results/baseline_results.txt", "w") as f:
        f.write("===== BASELINE RESULTS =====\n")
        f.write(f"Number of test runs: {num_runs}\n")
        f.write(
            f"Average wait time: {avg_wait_time_mean:.2f} "
            f"[{avg_wait_time_ci[0]:.2f}, {avg_wait_time_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total unique vehicle throughput: {throughput_mean:.2f} "
            f"[{throughput_ci[0]:.2f}, {throughput_ci[1]:.2f}]\n"
        )
        f.write(
            f"Total wait time: {total_wait_time_mean:.2f} "
            f"[{total_wait_time_ci[0]:.2f}, {total_wait_time_ci[1]:.2f}]\n"
        )

    return {
        "avg_wait_time": {
            "mean": avg_wait_time_mean,
            "ci_lower": avg_wait_time_ci[0],
            "ci_upper": avg_wait_time_ci[1],
        },
        "throughput": {
            "mean": throughput_mean,
            "ci_lower": throughput_ci[0],
            "ci_upper": throughput_ci[1],
        },
        "total_wait_time": {
            "mean": total_wait_time_mean,
            "ci_lower": total_wait_time_ci[0],
            "ci_upper": total_wait_time_ci[1],
        },
        "raw_data": {
            "avg_wait_times": avg_wait_times,
            "throughputs": throughputs,
            "total_wait_times": total_wait_times,
        },
        # --- NEW ---------------------------------------------------
        "per_step": {"avg": per_step_avg_runs, "total": per_step_total_runs},
        # -----------------------------------------------------------
    }


# ---------------------------------------------------------------------
#  COMPARISON FUNCTION
# ---------------------------------------------------------------------
def compare_all_results(
    baseline_results, webster_results, max_pressure_results, rl_results
):
    baseline_avg_wait = baseline_results["avg_wait_time"]["mean"]
    baseline_throughput = baseline_results["throughput"]["mean"]
    baseline_total_wait = baseline_results["total_wait_time"]["mean"]

    webster_avg_wait = webster_results["avg_wait_time"]["mean"]
    webster_throughput = webster_results["throughput"]["mean"]
    webster_total_wait = webster_results["total_wait_time"]["mean"]

    mp_avg_wait = max_pressure_results["avg_wait_time"]["mean"]
    mp_throughput = max_pressure_results["throughput"]["mean"]
    mp_total_wait = max_pressure_results["total_wait_time"]["mean"]

    rl_avg_wait = rl_results["avg_wait_time"]["mean"]
    rl_throughput = rl_results["throughput"]["mean"]
    rl_total_wait = rl_results["total_wait_time"]["mean"]

    webster_wait_improvement = (
        (baseline_avg_wait - webster_avg_wait) / baseline_avg_wait * 100
        if baseline_avg_wait > 0
        else 0
    )
    webster_throughput_improvement = (
        (webster_throughput - baseline_throughput) / baseline_throughput * 100
        if baseline_throughput > 0
        else 0
    )
    webster_total_wait_improvement = (
        (baseline_total_wait - webster_total_wait) / baseline_total_wait * 100
        if baseline_total_wait > 0
        else 0
    )

    mp_wait_improvement = (
        (baseline_avg_wait - mp_avg_wait) / baseline_avg_wait * 100
        if baseline_avg_wait > 0
        else 0
    )
    mp_throughput_improvement = (
        (mp_throughput - baseline_throughput) / baseline_throughput * 100
        if baseline_throughput > 0
        else 0
    )
    mp_total_wait_improvement = (
        (baseline_total_wait - mp_total_wait) / baseline_total_wait * 100
        if baseline_total_wait > 0
        else 0
    )

    rl_wait_improvement = (
        (baseline_avg_wait - rl_avg_wait) / baseline_avg_wait * 100
        if baseline_avg_wait > 0
        else 0
    )
    rl_throughput_improvement = (
        (rl_throughput - baseline_throughput) / baseline_throughput * 100
        if baseline_throughput > 0
        else 0
    )
    rl_total_wait_improvement = (
        (baseline_total_wait - rl_total_wait) / baseline_total_wait * 100
        if baseline_total_wait > 0
        else 0
    )

    print("\n===== COMPARISON OF ALL METHODS =====")
    print("\n--- Average Wait Time ---")
    print(
        f"Baseline: {baseline_avg_wait:.2f} "
        f"[{baseline_results['avg_wait_time']['ci_lower']:.2f}, "
        f"{baseline_results['avg_wait_time']['ci_upper']:.2f}]"
    )
    print(
        f"Webster: {webster_avg_wait:.2f} "
        f"[{webster_results['avg_wait_time']['ci_lower']:.2f}, "
        f"{webster_results['avg_wait_time']['ci_upper']:.2f}] "
        f"({webster_wait_improvement:.2f}% vs baseline)"
    )
    print(
        f"Max Pressure: {mp_avg_wait:.2f} "
        f"[{max_pressure_results['avg_wait_time']['ci_lower']:.2f}, "
        f"{max_pressure_results['avg_wait_time']['ci_upper']:.2f}] "
        f"({mp_wait_improvement:.2f}% vs baseline)"
    )
    print(
        f"RL Model: {rl_avg_wait:.2f} "
        f"[{rl_results['avg_wait_time']['ci_lower']:.2f}, "
        f"{rl_results['avg_wait_time']['ci_upper']:.2f}] "
        f"({rl_wait_improvement:.2f}% vs baseline)"
    )

    os.makedirs("comparison_plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    labels = ["Baseline", "Webster", "Max Pressure", "RL Model"]
    means = [baseline_avg_wait, webster_avg_wait, mp_avg_wait, rl_avg_wait]
    errors_lower = [
        baseline_avg_wait - baseline_results["avg_wait_time"]["ci_lower"],
        webster_avg_wait - webster_results["avg_wait_time"]["ci_lower"],
        mp_avg_wait - max_pressure_results["avg_wait_time"]["ci_lower"],
        rl_avg_wait - rl_results["avg_wait_time"]["ci_lower"],
    ]
    errors_upper = [
        baseline_results["avg_wait_time"]["ci_upper"] - baseline_avg_wait,
        webster_results["avg_wait_time"]["ci_upper"] - webster_avg_wait,
        max_pressure_results["avg_wait_time"]["ci_upper"] - mp_avg_wait,
        rl_results["avg_wait_time"]["ci_upper"] - rl_avg_wait,
    ]

    plt.bar(labels, means, yerr=[errors_lower, errors_upper], capsize=10)
    plt.ylabel("Average accumulated waiting time [s]")
    plt.title("Final accumulated waiting time per vehicle - all vehicles (mean ±95 % CI)")
    plt.savefig("comparison_plots/avg_wait_time_comparison.png")

    return {
        "webster": {
            "wait_time_improvement": webster_wait_improvement,
            "throughput_improvement": webster_throughput_improvement,
            "total_wait_improvement": webster_total_wait_improvement,
        },
        "max_pressure": {
            "wait_time_improvement": mp_wait_improvement,
            "throughput_improvement": mp_throughput_improvement,
            "total_wait_improvement": mp_total_wait_improvement,
        },
        "rl": {
            "wait_time_improvement": rl_wait_improvement,
            "throughput_improvement": rl_throughput_improvement,
            "total_wait_improvement": rl_total_wait_improvement,
        },
    }


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("baseline_results", exist_ok=True)
    os.makedirs("webster_results", exist_ok=True)
    os.makedirs("max_pressure_results", exist_ok=True)
    os.makedirs("comparison_plots", exist_ok=True)

    junctions_to_monitor = ["392163460", "392170608", "392180247", "249969885"]
    model_name = "city_tsc1"
    num_runs = 50
    steps = 1000
    seed = 111

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Test started at: {current_time}")
    print(f"Testing model: {model_name}")
    print(f"Number of runs: {num_runs}")
    print(f"Steps per run: {steps}")
    print(f"Random seed: {seed}")

    cfg_path = "configuration.sumocfg"
    if not os.path.exists(cfg_path):
        print(f"ERROR: Configuration file not found at {os.path.abspath(cfg_path)}")
        sys.exit(1)

    print("\n===== RUNNING BASELINE (NO RL) =====")
    baseline_results = run_sumo_without_rl(
        junctions_to_monitor, steps=steps, num_runs=num_runs, seed=seed
    )

    print("\n===== RUNNING WEBSTER'S METHOD =====")
    webster_results = run_sumo_with_webster(
        junctions_to_monitor, steps=steps, num_runs=num_runs, seed=seed
    )

    print("\n===== RUNNING MAX PRESSURE =====")
    max_pressure_results = run_sumo_with_max_pressure(
        junctions_to_monitor, steps=steps, num_runs=num_runs, seed=seed
    )

    print("\n===== TESTING RL MODEL =====")
    rl_results = test_rl_model(
        junctions_to_monitor,
        model_name=model_name,
        steps=steps,
        num_runs=num_runs,
        seed=seed,
    )

    if (
        rl_results is not None
        and webster_results is not None
        and max_pressure_results is not None
    ):
        improvements = compare_all_results(
            baseline_results, webster_results, max_pressure_results, rl_results
        )

        methods = ["Webster", "Max Pressure", "RL"]
        wait_improvements = [
            improvements["webster"]["wait_time_improvement"],
            improvements["max_pressure"]["wait_time_improvement"],
            improvements["rl"]["wait_time_improvement"],
        ]

        best_method = methods[np.argmax(wait_improvements)]
        best_improvement = max(wait_improvements)

        print("\n===== CONCLUSION =====")
        print(
            f"Best method for reducing wait time: {best_method} "
            f"({best_improvement:.2f}% improvement)"
        )

        if best_method == "RL":
            print(
                "The RL model outperforms both traditional methods, "
                "demonstrating the effectiveness of learning‑based "
                "approaches for traffic signal control."
            )
        else:
            print(
                f"The {best_method} method outperforms other approaches "
                "in this traffic scenario."
            )
    else:
        print("\nCouldn't compare all results because some tests failed.")

    # -----------------------------------------------------------------
    #  SAVE PER‑STEP ARRAYS  →  visualise_results.py will read this
    # -----------------------------------------------------------------
    np.savez_compressed(
        "comparison_plots/per_step_results.npz",
        baseline_avg=np.array(baseline_results["per_step"]["avg"]),
        baseline_total=np.array(baseline_results["per_step"]["total"]),
        webster_avg=np.array(webster_results["per_step"]["avg"]),
        webster_total=np.array(webster_results["per_step"]["total"]),
        mp_avg=np.array(max_pressure_results["per_step"]["avg"]),
        mp_total=np.array(max_pressure_results["per_step"]["total"]),
        rl_avg=np.array(rl_results["per_step"]["avg"]),
        rl_total=np.array(rl_results["per_step"]["total"]),
         steps=np.arange(steps),
    )

    print(
        "\nRaw per-step arrays saved to "
        "comparison_plots/per_step_results.npz is ready for visualisation."
    )
