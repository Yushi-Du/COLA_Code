# COLA: Learning Human-Humanoid Coordination for Collaborative Object Carrying

[![arxiv](https://img.shields.io/badge/arXiv%202510.14293-red?logo=arxiv)](https://arxiv.org/abs/2510.14293)
[![website](https://img.shields.io/badge/Project-0065D3?logo=rocket&logoColor=white)](https://yushi-du.github.io/COLA/)

![Logo](images/teaser.png)

COLA is a three-phase reinforcement-learning pipeline for collaborative object
carrying with a Unitree G1 humanoid:

1. whole-body locomotion pretraining;
2. privileged collaboration-teacher training;
3. student distillation for deployment.

## News

- Our code for the official whole pipeline is released! Thank you very much for waiting and we will continue to maintain this repo with more functions and updates. Check our newest [Live Demo](https://yushi-du.github.io/COLA/demo/wasm/) for better experience.

## Installation

The validated setup is Linux x86-64, Python 3.11, Isaac Sim 5.1.0, PyTorch
2.7.0, and a recent NVIDIA driver. Isaac Sim's pip distribution requires
GLIBC 2.35 or newer.

```bash
git clone --recurse-submodules https://github.com/Yushi-Du/COLA_Code.git
cd COLA_Code

conda create -n cola python=3.11 -y
conda activate cola
python -m pip install --upgrade pip

pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

./IsaacLab/isaaclab.sh --install
pip install -r requirements.txt
pip install -e rsl_rl
pip install -e .

source setup_env.sh
```

Run the setup script in every new shell after activating the environment. It
sets the release root, accepts the Isaac Sim EULA, and defaults W&B to offline
mode. To log online, run `export WANDB_MODE=online` before sourcing it.

## Training

- `train_locomotion.py`: train the Phase-1 locomotion policy.
- `train_collaboration.py`: train the Phase-2 teacher or Phase-3 student.
- `evaluate_locomotion.py`: evaluate a Phase-1 checkpoint.
- `evaluate_collaboration.py`: evaluate a Phase-2 or Phase-3 checkpoint.

The scripts are located in `legged_lab/scripts`. Complete commands, checkpoint
transitions, distributed topology, task names, and evaluation options are in
[docs/training.md](docs/training.md).

## MuJoCo sim2sim

Run the bundled Phase-3 student in the self-contained, robot-only MuJoCo scene:

```bash
python deployment/mujoco/run_sim2sim.py
```

To evaluate another Phase-1 actor or Phase-3 student, export and select it:

```bash
python deployment/mujoco/export_policy.py \
  --checkpoint /path/to/model_XXXX.pt \
  --output /path/to/policy.jit

python deployment/mujoco/run_sim2sim.py \
  --policy /path/to/policy.jit
```

For the current mass-conditioned student, optionally pass the measured object
mass with `--mass-observation-kg`; omitting it reproduces the episode-fixed
no-object training distribution.

See [deployment/mujoco/README.md](deployment/mujoco/README.md) for command
ranges, headless checks, and the physics contract.

## Citation

If you are interested in using our work, please cite:

```bibtex
@article{du2025learning,
  title={Learning Human-Humanoid Coordination for Collaborative Object Carrying},
  author={Yushi Du and Yixuan Li and Baoxiong Jia and Yutang Lin and Pei Zhou and Wei Liang and Yanchao Yang and Siyuan Huang},
  journal={arXiv preprint arXiv:2510.14293},
  year={2025}
}
```
