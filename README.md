# Project 2.1 - Machine Learning in the Unity Game Engine

Computer Science course project at Maastricht University. We collect data from Unity ML-Agents training runs and build machine learning models to predict training time, RAM usage, and agent robustness.

## Research Questions

- How accurately can training duration be predicted before a run starts?
- How accurately can peak and average RAM usage be predicted, and which parameters matter most?
- Can an agent's robustness to environmental changes be predicted from its training configuration and statistics?

This repository is a fork of Unity ML-Agents. It includes the Unity environments, Python trainers, and configuration files used as the foundation for our experiments.

## Requirements

- Git and Unity Hub.
- **Unity 2022.3.4f1**.
- **Python 3.10.1–3.10.12**; Python **3.10.11** is suitable for Windows.

## Quick Start (Windows PowerShell)

### 1. Install the Python Tools

```powershell
git clone https://github.com/ZeroV4/BCS2-ml-agents-fork.git
cd BCS2-ml-agents-fork

py -3.10 -m venv venv
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install "setuptools<81" wheel
.\venv\Scripts\python.exe -m pip install --no-build-isolation -e ./ml-agents-envs -e ./ml-agents
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\mlagents-learn.exe --help
```

Confirm that the printed Python version is within the supported range. If `py` is unavailable, replace `py -3.10` with the full path to your Python 3.10 interpreter. These commands use the virtual environment directly, so activation is unnecessary. The setuptools limit preserves `pkg_resources`, which this version of ML-Agents requires.

### 2. Open the Unity Scene

1. In Unity Hub, add the repository's **`Project`** folder and open it with Unity **2022.3.4f1**.
2. Wait for package installation and asset import to finish; resolve any Console compilation errors before continuing.
3. Open `Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity`.

The project already references the ML-Agents packages in this repository through local paths. Keep the repository's folder structure intact.

### 3. Start Training

From the repository root, run:

```powershell
.\venv\Scripts\mlagents-learn.exe config/ppo/3DBall.yaml --run-id=3DBall-first --seed=42
```

When the terminal says **Listening on port ...**, click **Play** in Unity. Training should then print reward statistics as it progresses. The example uses Behavior Name `3DBall` and Behavior Type `Default`, matching the training configuration.

Use a new `--run-id` for each experiment. To continue a run with saved checkpoints, add `--resume` and keep its original run ID.

### 4. View Training Metrics

In a second PowerShell terminal, from the repository root:

```powershell
.\venv\Scripts\tensorboard.exe --logdir results
```

Open the address shown in the terminal. Training outputs are stored in `results/<run-id>/`.

## Reproducing Experiments

Record the repository commit, dependency versions, hardware, configuration, seed, and command for every run. Change experimental settings through configuration files or command-line arguments. Document data preparation, dataset splits, and evaluation metrics alongside predictive-model results.

## Acknowledgments and License

Based on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). See [LICENSE.md](LICENSE.md).
