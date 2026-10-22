"""Run one ML-Agents training and append its features and targets to data/runs.csv.

Starts mlagents-learn, measures the RAM of the trainer and its child processes
(the Unity build) every second, and reads the configuration, TensorBoard
statistics and console output after the run.

Example:
    python data_collection/collect_run.py config/ppo/3DBall.yaml \
        --run-id 3dball_ppo_default --env ../BCS2-builds/3DBall/3DBall.x86_64
"""
import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import psutil
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_COLUMNS = [
    "run_id", "environment", "observation_size", "visual_observations", "num_agents", "num_areas",
    "trainer_type", "batch_size", "buffer_size", "learning_rate", "num_epoch", "hidden_units", "num_layers",
    "time_horizon", "max_steps", "use_memory",
    "num_envs", "time_scale", "no_graphics", "device",
    "cpu_model", "cpu_threads", "total_ram_mb", "gpu", "operating_system", "mlagents_version", "torch_version",
    "training_mass_min", "training_mass_max", "training_scale_min", "training_scale_max",
    "final_mean_reward", "final_reward_std", "reward_threshold", "steps_to_reward_threshold", "final_policy_entropy",
    "training_time_s", "peak_ram_mb", "average_ram_mb",
    "robustness_score", "robustness_score_median",
]

# ponytail: per-environment facts typed in by hand; read them from the behavior spec once we use many environments
ENVIRONMENT_INFO = {"3DBall": dict(observation_size=8, visual_observations=0, num_agents=12)}
# "Benchmark Mean Reward" from docs/Learning-Environment-Examples.md; the threshold is 90% of it
BENCHMARK_MEAN_REWARD = {"3DBall": 100.0}
THRESHOLD_FRACTION = 0.9


def run_training(command, log_path, sample_interval_s):
    """Run the command and return (wall-clock seconds, list of RAM samples in MB)."""
    ram_samples_mb = []
    start_time = time.time()
    with open(log_path, "w") as log_file:
        trainer_process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        trainer = psutil.Process(trainer_process.pid)
        while trainer_process.poll() is None:
            total_bytes = 0
            try:
                processes = [trainer] + trainer.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            for process in processes:
                try:
                    total_bytes += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass  # a child exited between listing and reading
            ram_samples_mb.append(total_bytes / 2**20)
            time.sleep(sample_interval_s)
    training_time_s = time.time() - start_time
    if trainer_process.returncode != 0:
        sys.exit(f"mlagents-learn failed with exit code {trainer_process.returncode}, see {log_path}")
    return training_time_s, ram_samples_mb


def parameter_range(environment_parameters, name):
    """Return (min, max) of an environment parameter in the resolved configuration, (1, 1) if not randomized."""
    found = {}

    def search(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("min_value", "max_value", "value") and isinstance(value, (int, float)):
                    found.setdefault(key, float(value))
                search(value)
        elif isinstance(node, list):
            for item in node:
                search(item)

    search(environment_parameters.get(name))
    if "min_value" in found:
        return found["min_value"], found["max_value"]
    constant = found.get("value", 1.0)
    return constant, constant


def cpu_model():
    if platform.system() == "Linux":
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="trainer configuration YAML")
    parser.add_argument("--run-id", required=True, help="new, unique run ID (runs are never resumed)")
    parser.add_argument("--env", required=True, help="path to the standalone environment build")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-file", default="data/runs.csv")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="seconds between RAM samples")
    args = parser.parse_args()

    run_folder = Path(args.results_dir) / args.run_id
    if run_folder.exists():
        sys.exit(f"{run_folder} already exists; use a new run ID")
    Path(args.results_dir).mkdir(exist_ok=True)
    log_path = Path(args.results_dir) / f"{args.run_id}_console.log"
    command = [sys.executable, "-m", "mlagents.trainers.learn", args.config, f"--run-id={args.run_id}",
               f"--env={args.env}", "--no-graphics", f"--results-dir={args.results_dir}"]
    training_time_s, ram_samples_mb = run_training(command, log_path, args.sample_interval)

    # configuration as ML-Agents resolved it, including defaults
    configuration = yaml.safe_load(open(run_folder / "configuration.yaml"))
    behavior_name, behavior = next(iter(configuration["behaviors"].items()))
    hyperparameters = behavior["hyperparameters"]
    network_settings = behavior["network_settings"]
    environment_parameters = configuration.get("environment_parameters") or {}
    mass_min, mass_max = parameter_range(environment_parameters, "mass")
    scale_min, scale_max = parameter_range(environment_parameters, "scale")
    device = configuration["torch_settings"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

    # last summary line printed by ML-Agents, e.g. "Mean Reward: 100.000. Std of Reward: 0.000."
    reward_lines = re.findall(r"Mean Reward: (-?\d+\.\d+)\. Std of Reward: (-?\d+\.\d+)", log_path.read_text())
    final_mean_reward, final_reward_std = reward_lines[-1] if reward_lines else ("", "")

    statistics = EventAccumulator(str(run_folder / behavior_name))
    statistics.Reload()
    reward_threshold = THRESHOLD_FRACTION * BENCHMARK_MEAN_REWARD[behavior_name]
    steps_to_threshold = next((event.step for event in statistics.Scalars("Environment/Cumulative Reward")
                               if event.value >= reward_threshold), "")

    row = dict(
        run_id=args.run_id, environment=behavior_name, num_areas=configuration["env_settings"]["num_areas"],
        **ENVIRONMENT_INFO[behavior_name],
        trainer_type=behavior["trainer_type"], batch_size=hyperparameters["batch_size"],
        buffer_size=hyperparameters["buffer_size"], learning_rate=hyperparameters["learning_rate"],
        num_epoch=hyperparameters.get("num_epoch", ""), hidden_units=network_settings["hidden_units"],
        num_layers=network_settings["num_layers"], time_horizon=behavior["time_horizon"],
        max_steps=behavior["max_steps"], use_memory=int(network_settings["memory"] is not None),
        num_envs=configuration["env_settings"]["num_envs"], time_scale=configuration["engine_settings"]["time_scale"],
        no_graphics=int(configuration["engine_settings"]["no_graphics"]), device=device,
        cpu_model=cpu_model(), cpu_threads=psutil.cpu_count(logical=True),
        total_ram_mb=round(psutil.virtual_memory().total / 2**20),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        operating_system=f"{platform.system()} {platform.release()}",
        mlagents_version=version("mlagents"), torch_version=torch.__version__.split("+")[0],
        training_mass_min=mass_min, training_mass_max=mass_max,
        training_scale_min=scale_min, training_scale_max=scale_max,
        final_mean_reward=final_mean_reward, final_reward_std=final_reward_std,
        reward_threshold=reward_threshold, steps_to_reward_threshold=steps_to_threshold,
        final_policy_entropy=round(statistics.Scalars("Policy/Entropy")[-1].value, 4),
        training_time_s=round(training_time_s, 1), peak_ram_mb=round(max(ram_samples_mb)),
        average_ram_mb=round(sum(ram_samples_mb) / len(ram_samples_mb)),
        robustness_score="", robustness_score_median="",  # filled in by evaluate_robustness.py
    )
    data_file = Path(args.data_file)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not data_file.exists()
    with open(data_file, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"{args.run_id}: {row['training_time_s']} s, peak {row['peak_ram_mb']} MB, "
          f"average {row['average_ram_mb']} MB, added to {data_file}")


if __name__ == "__main__":
    main()
