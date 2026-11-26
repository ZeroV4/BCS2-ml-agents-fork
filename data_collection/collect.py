"""
Examples (run from the repo root inside the mlagents environment):
    python data_collection/collect.py --runs 5
    python data_collection/collect.py --env PushBlock
    python data_collection/collect.py --env 3DBall --max-steps 20000 --no-git      # quick test
    python data_collection/collect.py --repeat-of data/Vadim/Vadim_run3.csv        # same config, new seed
"""

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median

import numpy as np
import psutil
import torch
import yaml
from mlagents.plugins.trainer_type import register_trainer_plugins
from mlagents.torch_utils import set_torch_config
from mlagents.trainers.settings import RunOptions, TorchSettings
from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECTION_CONFIG = REPO_ROOT / "data_collection" / "collection.yaml"
LOCAL_CONFIG = REPO_ROOT / "data_collection" / "local.yaml"
RESULTS_DIR = REPO_ROOT / "results"   # gitignored, ML-Agents writes its run folders here
DATA_DIR = REPO_ROOT / "data"
NETWORK_KEYS = {"hidden_units", "num_layers"}   # these go to network_settings, the rest to hyperparameters
EPISODE_COLUMNS = ["run_id", "test_parameter", "default_value", "test_value", "change_factor",
                   "agent_id", "episode_number", "eval_seed", "episode_reward", "episode_length"]


def load_settings():
    if not LOCAL_CONFIG.exists():
        sys.exit(f"{LOCAL_CONFIG} is missing: copy local.example.yaml to local.yaml and fill in your name and builds")
    shared = yaml.safe_load(COLLECTION_CONFIG.read_text())
    local = yaml.safe_load(LOCAL_CONFIG.read_text())
    member = str(local.get("member", "")).strip()
    # the name ends up in file names and run ids, so only plain letters, digits, ., - and _
    if not re.fullmatch(r"[A-Za-z0-9._-]+", member) or member == "YourName":
        sys.exit(f"set 'member' in {LOCAL_CONFIG} to your name (letters, digits, ., - or _ only)")
    builds = {}
    for environment_name, build_path in (local.get("builds") or {}).items():
        build_path = Path(build_path)
        build_path = build_path if build_path.is_absolute() else REPO_ROOT / build_path
        if environment_name in shared["environments"] and build_path.exists():
            builds[environment_name] = build_path.resolve()
    if not builds:
        sys.exit(f"none of the builds in {LOCAL_CONFIG} exist")
    return shared, member, builds


def next_run_number(member_folder, member):
    numbers = [int(match.group(1)) for path in member_folder.glob(f"{member}_run*.csv")
               if (match := re.fullmatch(rf"{re.escape(member)}_run(\d+)\.csv", path.name))]
    return max(numbers, default=0) + 1


def sample_value(rule, random_generator):
    if "choice" in rule:
        return random_generator.choice(rule["choice"])
    low, high = rule["log_uniform"]
    # log-uniform, so 0.0001-0.0002 is picked as often as 0.0005-0.001
    return round(float(np.exp(random_generator.uniform(np.log(low), np.log(high)))), 7)


def sample_choices(environment_name, environment, shared, random_generator, gpu_available):
    trainer_type = random_generator.choice(sorted(set(environment["trainers"]) & set(shared["trainers"])))
    hyperparameters = {key: sample_value(rule, random_generator) for key, rule in shared["trainers"][trainer_type].items()}
    randomization = {}
    if random_generator.random() < shared["randomization_probability"]:
        # every parameter gets its own range around the default, so the width of the range varies between runs
        for parameter, details in environment["parameters"].items():
            lowest, highest = details["randomize"]
            randomization[parameter] = [round(random_generator.uniform(lowest, details["default"]), 3),
                                        round(random_generator.uniform(details["default"], highest), 3)]
    use_gpu = gpu_available and random_generator.random() < shared["gpu_probability"]
    return {
        "environment": environment_name,
        "trainer_type": trainer_type,
        "max_steps": random_generator.randint(*environment["max_steps"]),
        "num_envs": random_generator.choice(shared["num_envs"]),
        "device": "cuda" if use_gpu else "cpu",
        "hyperparameters": hyperparameters,
        "randomization": randomization,
    }


def choices_from_row(row, shared):
    """Rebuild the configuration of an earlier run, to repeat it with a new seed (measures the noise floor)."""
    environment = shared["environments"][row["environment"]]
    parse = yaml.safe_load   # turns "0.0003" into 0.0003 and "64" into 64
    randomization = {}
    for parameter in environment["parameters"]:
        lowest, highest = parse(row[f"train_{parameter}_min"]), parse(row[f"train_{parameter}_max"])
        if lowest != highest:
            randomization[parameter] = [lowest, highest]
    return {
        "environment": row["environment"],
        "trainer_type": row["trainer_type"],
        "max_steps": parse(row["max_steps"]),
        "num_envs": parse(row["num_envs"]),
        "device": row["device"],
        "hyperparameters": {key: parse(row[key]) for key in shared["trainers"][row["trainer_type"]]},
        "randomization": randomization,
    }


def config_id(choices):
    # identical configurations get the same id, so repeats stay in the same cross-validation fold later
    return hashlib.sha1(json.dumps(choices, sort_keys=True).encode()).hexdigest()[:10]


def write_trainer_config(choices, environment, config_path):
    trainer_config = yaml.safe_load((REPO_ROOT / environment["trainers"][choices["trainer_type"]]).read_text())
    behavior = trainer_config["behaviors"][environment["behavior_name"]]
    for key, value in choices["hyperparameters"].items():
        section = "network_settings" if key in NETWORK_KEYS else "hyperparameters"
        behavior[section][key] = value
    behavior["max_steps"] = choices["max_steps"]
    trainer_config.pop("environment_parameters", None)
    if choices["randomization"]:
        trainer_config["environment_parameters"] = {
            parameter: {"sampler_type": "uniform", "sampler_parameters": {"min_value": lowest, "max_value": highest}}
            for parameter, (lowest, highest) in choices["randomization"].items()}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(trainer_config, sort_keys=False))
    return behavior


class RamMonitor:
    """Samples the RAM of a process and all its children in a background thread.

    Processes started from the environment build count as environment RAM, all others as trainer RAM.
    RSS counts shared libraries in every process, so totals are a bit high, but equally high in every run.
    """

    def __init__(self, root_process_id, build_path, interval_s):
        self.root_process = psutil.Process(root_process_id)
        self.build_name = Path(build_path).stem.lower()
        self.interval_s = interval_s
        self.trainer_samples_mb, self.environment_samples_mb = [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            trainer_bytes = environment_bytes = 0
            try:
                processes = [self.root_process] + self.root_process.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            for process in processes:
                try:
                    command = process.cmdline()
                    resident_bytes = process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue   # a process ended between listing and reading
                if command and Path(command[0]).stem.lower() == self.build_name:
                    environment_bytes += resident_bytes
                else:
                    trainer_bytes += resident_bytes
            self.trainer_samples_mb.append(trainer_bytes / 2**20)
            self.environment_samples_mb.append(environment_bytes / 2**20)
            self.stop_event.wait(self.interval_s)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def summary(self, prefix, split=True):
        total_samples = [trainer + environment for trainer, environment
                         in zip(self.trainer_samples_mb, self.environment_samples_mb)]
        groups = [("", total_samples)]
        if split:
            groups += [("trainer_", self.trainer_samples_mb), ("environment_", self.environment_samples_mb)]
        result = {}
        for name, samples in groups:
            result[f"{prefix}{name}peak_ram_mb"] = round(max(samples)) if samples else ""
            result[f"{prefix}{name}average_ram_mb"] = round(mean(samples)) if samples else ""
        return result


def train(config_path, results_run_id, build_path, choices, seed, needs_graphics, interval_s):
    log_path = RESULTS_DIR / f"{results_run_id}_console.log"
    command = [sys.executable, "-m", "mlagents.trainers.learn", str(config_path),
               f"--run-id={results_run_id}", f"--env={build_path}", f"--results-dir={RESULTS_DIR}",
               f"--seed={seed}", f"--num-envs={choices['num_envs']}", f"--torch-device={choices['device']}"]
    if not needs_graphics:
        command.append("--no-graphics")
    start_time = time.time()
    with open(log_path, "w") as log_file:
        trainer_process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        monitor = RamMonitor(trainer_process.pid, build_path, interval_s)
        monitor.start()
        return_code = trainer_process.wait()
        monitor.stop()
    if return_code != 0:
        raise RuntimeError(f"mlagents-learn failed with exit code {return_code}, see {log_path}")
    return {"training_time_s": round(time.time() - start_time, 1), **monitor.summary("")}, log_path


def read_training_statistics(run_folder, behavior_name, log_path, reward_threshold):
    # last summary line printed by ML-Agents, e.g. "Mean Reward: 100.000. Std of Reward: 0.000."
    reward_lines = re.findall(r"Mean Reward: (-?\d+\.\d+)\. Std of Reward: (-?\d+\.\d+)", log_path.read_text())
    final_mean_reward, final_reward_std = reward_lines[-1] if reward_lines else ("", "")
    statistics = EventAccumulator(str(run_folder / behavior_name))
    statistics.Reload()
    scalar_tags = statistics.Tags()["scalars"]
    reward_curve = statistics.Scalars("Environment/Cumulative Reward") if "Environment/Cumulative Reward" in scalar_tags else []
    entropy_curve = statistics.Scalars("Policy/Entropy") if "Policy/Entropy" in scalar_tags else []
    return {
        "final_mean_reward": final_mean_reward,
        "final_reward_std": final_reward_std,
        "reward_threshold": reward_threshold,
        # TensorBoard averages over each summary period, so this is exact up to one summary_freq
        "steps_to_reward_threshold": next((event.step for event in reward_curve if event.value >= reward_threshold), ""),
        "final_policy_entropy": round(entropy_curve[-1].value, 4) if entropy_curve else "",
    }


def choose_actions(actor, decision_steps, action_spec):
    observations = [torch.as_tensor(observation) for observation in decision_steps.obs]
    masks = None
    if action_spec.discrete_size > 0:
        # same mask as ML-Agents uses in training (TorchPolicy._extract_masks), e.g. GridWorld forbids walking into walls
        masks = torch.ones(len(decision_steps), int(np.sum(action_spec.discrete_branches)))
        if decision_steps.action_mask is not None:
            masks = torch.as_tensor(1 - np.concatenate(decision_steps.action_mask, axis=1), dtype=torch.float32)
    with torch.no_grad():
        outputs = actor(observations, masks=masks)
    # SimpleActor.forward returns [version, memory size, (continuous, size, deterministic continuous),
    # (discrete, size, deterministic discrete)], where a group is only there if the agent has that action type.
    # We take the deterministic (most likely) actions, the same outputs Unity uses for inference.
    continuous_actions = discrete_actions = None
    output_index = 2
    if action_spec.continuous_size > 0:
        continuous_actions = outputs[output_index + 2].cpu().numpy()
        output_index += 3
    if action_spec.discrete_size > 0:
        discrete_actions = outputs[output_index + 2].cpu().numpy().astype(np.int32)
    return ActionTuple(continuous=continuous_actions, discrete=discrete_actions)


def play_episodes(environment, behavior_name, actor, action_spec, episodes_per_agent):
    """Play until every agent finished episodes_per_agent episodes; return {agent_id: [(reward, length), ...]}.

    We take the first episodes of every agent, because failed episodes end sooner and would otherwise
    be counted more often.

    Reward and length are counted like ML-Agents does in training (AgentProcessor): a step only counts
    once the agent has acted in the episode, and the last step adds reward but no length. So a full
    3DBall episode gives 100.0 and 999, the same numbers as in TensorBoard.
    """
    environment.reset()
    current_reward, current_length, finished_episodes = defaultdict(float), defaultdict(int), defaultdict(list)
    has_acted = defaultdict(bool)
    seen_agents = set()
    while not seen_agents or min(len(finished_episodes[agent_id]) for agent_id in seen_agents) < episodes_per_agent:
        decision_steps, terminal_steps = environment.get_steps(behavior_name)
        for agent_id in terminal_steps.agent_id:
            seen_agents.add(agent_id)
            final_reward = terminal_steps[agent_id].reward if has_acted[agent_id] else 0.0
            finished_episodes[agent_id].append((current_reward[agent_id] + final_reward, current_length[agent_id]))
            current_reward[agent_id], current_length[agent_id], has_acted[agent_id] = 0.0, 0, False
        for agent_id in decision_steps.agent_id:
            seen_agents.add(agent_id)
            if has_acted[agent_id]:   # the first decision of an episode follows no action of ours
                current_reward[agent_id] += decision_steps[agent_id].reward
                current_length[agent_id] += 1
            has_acted[agent_id] = True   # it gets an action below
        if len(decision_steps) > 0:
            environment.set_actions(behavior_name, choose_actions(actor, decision_steps, action_spec))
        environment.step()
    return {agent_id: finished_episodes[agent_id][:episodes_per_agent] for agent_id in seen_agents}


def evaluate(run_id, run_folder, environment_settings, build_path, shared, reward_threshold):
    """Test the trained agent in the normal environment and, if it learned the task, in every changed one."""
    evaluation = shared["evaluation"]
    behavior_name = environment_settings["behavior_name"]
    parameters = environment_settings["parameters"]
    register_trainer_plugins()   # RunOptions needs the PPO/SAC settings classes, as in mlagents-learn
    set_torch_config(TorchSettings(device="cpu"))   # evaluation always on the CPU, only the training device varies
    run_options = RunOptions.from_dict(yaml.safe_load((run_folder / "configuration.yaml").read_text()))
    behavior_settings = run_options.behaviors[behavior_name]
    is_sac = behavior_settings.trainer_type == "sac"

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=run_options.engine_settings.time_scale)
    parameters_channel = EnvironmentParametersChannel()
    start_time = time.time()
    monitor = RamMonitor(psutil.Process().pid, build_path, shared["ram_sample_interval_s"])
    monitor.start()
    episode_rows, rewards_per_test = [], {}
    try:
        environment = UnityEnvironment(file_name=str(build_path), worker_id=1, seed=evaluation["seed"],
                                       no_graphics=not environment_settings["needs_graphics"],
                                       side_channels=[engine_channel, parameters_channel])
        try:
            environment.reset()
            full_behavior_name = next(iter(environment.behavior_specs))
            behavior_spec = environment.behavior_specs[full_behavior_name]
            actor = SimpleActor(behavior_spec.observation_specs, behavior_settings.network_settings,
                                behavior_spec.action_spec, conditional_sigma=is_sac, tanh_squash=is_sac)
            checkpoint = torch.load(run_folder / behavior_name / "checkpoint.pt", map_location="cpu")
            actor.load_state_dict(checkpoint["Policy"])
            actor.eval()

            tests = [(None, None)] + [(parameter, value) for parameter, details in parameters.items()
                                      for value in details["tests"]]
            agent_count = 0
            for test_parameter, test_value in tests:
                # set every parameter each time, so a test never inherits the change of the test before it
                for parameter, details in parameters.items():
                    parameters_channel.set_float_parameter(
                        parameter, float(test_value if parameter == test_parameter else details["default"]))
                episodes = play_episodes(environment, full_behavior_name, actor, behavior_spec.action_spec,
                                         evaluation["episodes_per_agent"])
                agent_count = max(agent_count, len(episodes))
                rewards_per_test[(test_parameter, test_value)] = [reward for agent_episodes in episodes.values()
                                                                  for reward, _ in agent_episodes]
                default_value = parameters[test_parameter]["default"] if test_parameter else ""
                if not test_parameter:
                    change_factor = 1.0
                elif default_value:
                    change_factor = round(test_value / default_value, 3)
                else:
                    change_factor = ""   # the default is 0 (friction), so a ratio makes no sense
                for agent_id, agent_episodes in episodes.items():
                    for episode_number, (reward, length) in enumerate(agent_episodes, start=1):
                        episode_rows.append({
                            "run_id": run_id, "test_parameter": test_parameter or "none",
                            "default_value": default_value, "test_value": test_value if test_parameter else "",
                            "change_factor": change_factor, "agent_id": agent_id, "episode_number": episode_number,
                            "eval_seed": evaluation["seed"], "episode_reward": round(reward, 3),
                            "episode_length": length})
                # retention of an agent that never learned the task says nothing about robustness, so stop here
                if test_parameter is None and mean(rewards_per_test[(None, None)]) < reward_threshold:
                    break
        finally:
            environment.close()
    finally:
        monitor.stop()

    baseline_rewards = rewards_per_test.pop((None, None))
    observation_specs = behavior_spec.observation_specs
    summary = {
        "observation_size": sum(int(np.prod(spec.shape)) for spec in observation_specs),
        "visual_observations": int(any(len(spec.shape) == 3 for spec in observation_specs)),
        "action_continuous_size": behavior_spec.action_spec.continuous_size,
        "action_discrete_branches": len(behavior_spec.action_spec.discrete_branches),
        "num_agents": agent_count,
        "baseline_mean_reward": round(mean(baseline_rewards), 3),
        "robustness_evaluated": int(bool(rewards_per_test)),
        "robustness_skip_reason": "" if rewards_per_test else "baseline reward below threshold",
        "robustness_score": round(mean(mean(rewards) / mean(baseline_rewards)
                                       for rewards in rewards_per_test.values()), 3) if rewards_per_test else "",
        "robustness_score_median": round(mean(median(rewards) / median(baseline_rewards)
                                              for rewards in rewards_per_test.values()), 3)
                                   if rewards_per_test and median(baseline_rewards) else "",
        "eval_episodes_per_agent": evaluation["episodes_per_agent"],
        "eval_seed": evaluation["seed"],
        "eval_duration_s": round(time.time() - start_time, 1),
        **monitor.summary("eval_", split=False),
    }
    return summary, episode_rows


def cpu_model():
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def gpu_name():
    # asked in a separate process: reading the name starts the GPU driver, which would otherwise
    # stay in this process and inflate the RAM we measure during the evaluation
    answer = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0) "
                             "if torch.cuda.is_available() else '')"], capture_output=True, text=True)
    return answer.stdout.strip()


def hardware_info(gpu):
    frequency = psutil.cpu_freq()
    return {
        "cpu_model": cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_max_frequency_mhz": round(frequency.max) if frequency and frequency.max else "",
        "total_ram_mb": round(psutil.virtual_memory().total / 2**20),
        "gpu": gpu or "none",
        "operating_system": f"{platform.system()} {platform.release()}",
        "mlagents_version": version("mlagents"),
        "torch_version": torch.__version__.split("+")[0],
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]) if rows else EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths, run_number, environment_name):
    """Commit only this run's files with the member's own git identity, then push the current branch."""
    relative_paths = [str(path.relative_to(REPO_ROOT)) for path in paths]
    git("add", "--", *relative_paths)
    # giving paths to git commit commits only those paths, even if other files are staged (git help commit, --only)
    # no "#" before the number: GitHub turns "#106" into a link to issue/pull request 106
    committed = git("commit", "-m", f"run {run_number}, {environment_name}", "--", *relative_paths)
    if committed.returncode != 0:
        print(f"warning: commit failed, the data is saved but not committed:\n{committed.stderr}{committed.stdout}")
        return
    if git("push", "origin", "HEAD").returncode == 0:
        return
    # someone else pushed first: put our commit on top of theirs and try once more
    branch = git("branch", "--show-current").stdout.strip()
    if git("pull", "--rebase", "--autostash", "origin", branch).returncode != 0:
        git("rebase", "--abort")   # never leave the repo half-rebased while running unattended
    pushed = git("push", "origin", "HEAD")
    if pushed.returncode != 0:
        print(f"warning: push failed, the commit stays local and goes out with the next push:\n{pushed.stderr}")


def run_once(environment_name, choices, repeat_of, max_steps_override, shared, member, builds, use_git, hardware):
    run_start = time.time()
    gpu_available = hardware["gpu"] != "none"
    environment_settings = shared["environments"][environment_name]
    behavior_name = environment_settings["behavior_name"]
    member_folder = DATA_DIR / member
    member_folder.mkdir(parents=True, exist_ok=True)
    run_number = next_run_number(member_folder, member)
    run_id = f"{member}_run{run_number}"
    # ML-Agents refuses to reuse a results folder, so a crashed earlier attempt must not block the name
    results_run_id = f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seed = secrets.randbelow(90000) + 10000   # drawn per run and stored, so the run can be reproduced
    if choices is None:
        choices = sample_choices(environment_name, environment_settings, shared, random.Random(seed), gpu_available)
    else:
        choices = copy.deepcopy(choices)   # the override below must not change the repeated configuration
    if max_steps_override:
        choices["max_steps"] = max_steps_override
    print(f"\n{run_id}: {environment_name}, {choices['trainer_type']}, {choices['max_steps']} steps, "
          f"{choices['num_envs']} env(s), {choices['device']}, randomized={bool(choices['randomization'])}")

    config_path = RESULTS_DIR / "configs" / f"{results_run_id}.yaml"
    behavior = write_trainer_config(choices, environment_settings, config_path)
    reward_threshold = round(shared["evaluation"]["threshold_fraction"] * environment_settings["benchmark_mean_reward"], 3)
    training, log_path = train(config_path, results_run_id, builds[environment_name], choices, seed,
                               environment_settings["needs_graphics"], shared["ram_sample_interval_s"])
    run_folder = RESULTS_DIR / results_run_id
    resolved_config = yaml.safe_load((run_folder / "configuration.yaml").read_text())   # as ML-Agents ran it
    training_statistics = read_training_statistics(run_folder, behavior_name, log_path, reward_threshold)
    print(f"  trained in {training['training_time_s']} s, peak RAM {training['peak_ram_mb']} MB, "
          f"final reward {training_statistics['final_mean_reward']}")
    try:
        evaluation, episode_rows = evaluate(run_id, run_folder, environment_settings, builds[environment_name],
                                            shared, reward_threshold)
    except Exception as error:   # keep the training data even if the evaluation breaks
        print(f"  warning: evaluation failed: {error}")
        evaluation, episode_rows = {"robustness_evaluated": 0, "robustness_skip_reason": f"evaluation failed: {error}"}, []

    row = {
        "run_id": run_id, "member": member, "run_number": run_number,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "config_id": config_id(choices), "seed": seed, "repeat_of": repeat_of or "",
        "environment": environment_name, "behavior_name": behavior_name,
        **{key: evaluation.pop(key, "") for key in ["observation_size", "visual_observations",
                                                    "action_continuous_size", "action_discrete_branches", "num_agents"]},
        "trainer_type": choices["trainer_type"], **choices["hyperparameters"],
        "time_horizon": behavior["time_horizon"], "max_steps": choices["max_steps"],
        "num_envs": choices["num_envs"], "time_scale": resolved_config["engine_settings"]["time_scale"],
        "no_graphics": int(not environment_settings["needs_graphics"]), "device": choices["device"],
        **hardware,
        "randomized": int(bool(choices["randomization"])),
    }
    for parameter, details in environment_settings["parameters"].items():
        lowest, highest = choices["randomization"].get(parameter, [details["default"], details["default"]])
        row[f"train_{parameter}_min"], row[f"train_{parameter}_max"] = lowest, highest
    row.update(training_statistics)
    row.update(training)
    row.update(evaluation)
    row["total_run_time_s"] = round(time.time() - run_start, 1)   # training + evaluation + reading, before saving

    run_file = member_folder / f"{run_id}.csv"
    episodes_file = member_folder / f"{run_id}_episodes.csv"
    write_csv(run_file, [row])
    write_csv(episodes_file, episode_rows)
    print(f"  saved {run_file.relative_to(REPO_ROOT)} (robustness score {row.get('robustness_score', '')}, "
          f"total {row['total_run_time_s']} s)")
    if use_git:
        commit_and_push([run_file, episodes_file], run_number, environment_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="number of runs to do one after another")
    parser.add_argument("--env", action="append", help="environment(s) to pick from (default: every environment with a build)")
    parser.add_argument("--max-steps", type=int, help="override the sampled max_steps, for quick tests only")
    parser.add_argument("--repeat-of", type=Path, help="run file of an earlier run to repeat with a new seed")
    parser.add_argument("--no-git", action="store_true", help="save the data but do not commit or push")
    args = parser.parse_args()

    shared, member, builds = load_settings()
    hardware = hardware_info(gpu_name())
    repeat_choices = None
    if args.repeat_of:
        with open(args.repeat_of, newline="") as csv_file:
            repeat_row = next(csv.DictReader(csv_file))
        repeat_choices = choices_from_row(repeat_row, shared)
        if repeat_choices["device"] == "cuda" and hardware["gpu"] == "none":
            sys.exit("that run trained on a GPU, but this machine has none")
        environment_names = [repeat_choices["environment"]]
    else:
        environment_names = args.env or sorted(builds)
    missing = [name for name in environment_names if name not in builds]
    if missing:
        sys.exit(f"no build for {missing} in {LOCAL_CONFIG}")

    for _ in range(args.runs):
        try:
            run_once(random.choice(environment_names), repeat_choices,
                     args.repeat_of.stem if args.repeat_of else None, args.max_steps,
                     shared, member, builds, not args.no_git, hardware)
        except RuntimeError as error:   # a failed training run is skipped, the loop goes on
            print(f"  run failed: {error}")


if __name__ == "__main__":
    main()
