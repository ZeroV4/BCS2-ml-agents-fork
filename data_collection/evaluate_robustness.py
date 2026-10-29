"""Test a trained 3DBall agent in changed environments (RQ3).

Loads the final checkpoint of a run, plays a fixed number of episodes per agent
in the normal environment and in every changed environment, writes one row per
episode to data/robustness_episodes.csv, and fills in the robustness scores of
the run in data/runs.csv.

Example:
    python data_collection/evaluate_robustness.py 3dball_ppo_default \
        --env ../BCS2-builds/3DBall/3DBall.x86_64
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch
import yaml
from mlagents.plugins.trainer_type import register_trainer_plugins
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

EPISODE_COLUMNS = ["run_id", "test_parameter", "default_value", "test_value", "change_factor",
                   "agent_id", "episode_number", "eval_seed", "episode_reward", "episode_length"]
DEFAULT_PARAMETERS = {"mass": 1.0, "scale": 1.0}
TEST_VALUES = {"mass": [0.5, 2.0, 5.0, 10.0], "scale": [0.75, 1.5, 2.0, 3.0]}


def play_episodes(environment, behavior_name, actor, episodes_per_agent):
    """Play until every agent has finished episodes_per_agent episodes; return {agent_id: [(reward, length), ...]}.

    Taking the first episodes of every agent avoids favouring short (failed) episodes, which finish first.
    """
    environment.reset()
    current_reward, current_length, finished = defaultdict(float), defaultdict(int), defaultdict(list)
    decision_steps, _ = environment.get_steps(behavior_name)
    agent_ids = list(decision_steps.agent_id)
    while min(len(finished[agent_id]) for agent_id in agent_ids) < episodes_per_agent:
        decision_steps, terminal_steps = environment.get_steps(behavior_name)
        for agent_id in terminal_steps.agent_id:
            finished[agent_id].append((current_reward[agent_id] + terminal_steps[agent_id].reward,
                                       current_length[agent_id] + 1))
            current_reward[agent_id], current_length[agent_id] = 0.0, 0
        for agent_id in decision_steps.agent_id:
            current_reward[agent_id] += decision_steps[agent_id].reward
            current_length[agent_id] += 1
        if len(decision_steps) > 0:
            with torch.no_grad():
                actor_outputs = actor([torch.as_tensor(observation) for observation in decision_steps.obs])
            deterministic_actions = actor_outputs[4]  # same output Unity uses for inference
            environment.set_actions(behavior_name, ActionTuple(continuous=deterministic_actions.cpu().numpy()))
        environment.step()
    return {agent_id: finished[agent_id][:episodes_per_agent] for agent_id in agent_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--env", required=True, help="path to the standalone environment build")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--episodes-per-agent", type=int, default=8, help="3DBall has 12 agents, so 8 gives 96 episodes")
    parser.add_argument("--seed", type=int, default=88280, help="same seed for every agent we test")
    parser.add_argument("--worker-id", type=int, default=1, help="use different ids to evaluate in parallel")
    parser.add_argument("--episodes-file", default="data/robustness_episodes.csv")
    parser.add_argument("--runs-file", default="data/runs.csv")
    args = parser.parse_args()

    run_folder = Path(args.results_dir) / args.run_id
    register_trainer_plugins()  # RunOptions needs the PPO/SAC settings classes, as in mlagents-learn
    run_options = RunOptions.from_dict(yaml.safe_load(open(run_folder / "configuration.yaml")))
    behavior_name, behavior_settings = next(iter(run_options.behaviors.items()))
    is_sac = behavior_settings.trainer_type == "sac"

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=run_options.engine_settings.time_scale)
    parameters_channel = EnvironmentParametersChannel()
    environment = UnityEnvironment(file_name=args.env, worker_id=args.worker_id, seed=args.seed, no_graphics=True,
                                   side_channels=[engine_channel, parameters_channel])
    environment.reset()
    full_behavior_name = next(iter(environment.behavior_specs))
    behavior_spec = environment.behavior_specs[full_behavior_name]
    actor = SimpleActor(behavior_spec.observation_specs, behavior_settings.network_settings,
                        behavior_spec.action_spec, conditional_sigma=is_sac, tanh_squash=is_sac)
    checkpoint = torch.load(run_folder / behavior_name / "checkpoint.pt", map_location="cpu")
    actor.load_state_dict(checkpoint["Policy"])
    actor.eval()

    tests = [(None, None)] + [(name, value) for name, values in TEST_VALUES.items() for value in values]
    episode_rows, rewards_per_test = [], {}
    try:
        for test_parameter, test_value in tests:
            parameters = dict(DEFAULT_PARAMETERS)
            if test_parameter:
                parameters[test_parameter] = test_value
            for name, value in parameters.items():
                parameters_channel.set_float_parameter(name, value)
            episodes = play_episodes(environment, full_behavior_name, actor, args.episodes_per_agent)
            rewards_per_test[(test_parameter, test_value)] = [reward for agent_episodes in episodes.values()
                                                              for reward, _ in agent_episodes]
            default_value = DEFAULT_PARAMETERS.get(test_parameter, "")
            for agent_id, agent_episodes in episodes.items():
                for episode_number, (reward, length) in enumerate(agent_episodes, start=1):
                    episode_rows.append(dict(
                        run_id=args.run_id, test_parameter=test_parameter or "none", default_value=default_value,
                        test_value=test_value if test_parameter else "",
                        change_factor=round(test_value / default_value, 3) if test_parameter else 1.0,
                        agent_id=agent_id, episode_number=episode_number, eval_seed=args.seed,
                        episode_reward=round(reward, 3), episode_length=length))
            print(f"{test_parameter or 'baseline'} {test_value or ''}: mean reward "
                  f"{mean(rewards_per_test[(test_parameter, test_value)]):.1f}")
    finally:
        environment.close()

    episodes_file = Path(args.episodes_file)
    write_header = not episodes_file.exists()
    with open(episodes_file, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EPISODE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(episode_rows)

    baseline_rewards = rewards_per_test.pop((None, None))
    robustness_score = mean(mean(rewards) / mean(baseline_rewards) for rewards in rewards_per_test.values())
    robustness_score_median = mean(median(rewards) / median(baseline_rewards) for rewards in rewards_per_test.values())
    with open(args.runs_file, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        run_columns, run_rows = reader.fieldnames, list(reader)
    for run_row in run_rows:
        if run_row["run_id"] == args.run_id:
            run_row["robustness_score"] = round(robustness_score, 3)
            run_row["robustness_score_median"] = round(robustness_score_median, 3)
    with open(args.runs_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=run_columns)
        writer.writeheader()
        writer.writerows(run_rows)
    print(f"{args.run_id}: robustness score {robustness_score:.3f} (median-based {robustness_score_median:.3f})")


if __name__ == "__main__":
    main()
