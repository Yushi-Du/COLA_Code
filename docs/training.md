# COLA training pipeline

Run all commands from the repository root after `conda activate cola` and
`source setup_env.sh`. The examples below use eight GPUs and 2048 environments
per GPU. Reduce `--num_envs` for smaller GPUs.

## Topology

Phase 2 and Phase 3 use the same static rank assignment:

| Ranks | Scene |
|---|---|
| 0-2 | Centered bar fixed to the left wrist |
| 3-5 | Centered bar fixed to the right wrist (mirrored) |
| 6-7 | Fixed-hand robot without a bar |

## Phase 1: locomotion

```bash
torchrun --standalone --nproc_per_node=8 \
  legged_lab/scripts/train_locomotion.py \
  --task=cola_phase_1_locomotion \
  --distributed --headless --num_envs=2048 --max_iterations=4000 \
  --logger=wandb --run_name=phase1_locomotion
```

Select the Phase-1 output for Phase 2:

```bash
export PHASE1_RUN=/absolute/path/to/logs/cola_phase_1_locomotion/RUN_DIRECTORY
export PHASE1_CHECKPOINT=model_3999.pt
```

## Phase 2: collaboration teacher

```bash
torchrun --standalone --nproc_per_node=8 \
  legged_lab/scripts/train_collaboration.py \
  --phase=2 \
  --distributed --headless --num_envs=2048 --max_iterations=8000 \
  --no_object_rank_count=2 --right_fixed_rank_count=3 \
  --resume=True --load_run="$PHASE1_RUN" --checkpoint="$PHASE1_CHECKPOINT" \
  --logger=wandb --run_name=phase2_teacher
```

Select the Phase-2 output for Phase 3:

```bash
export PHASE2_RUN="$(pwd)/logs/cola_phase_2_teacher_left_fixed_bar/static_mix_phase2_teacher"
export PHASE2_CHECKPOINT=model_7999.pt
```

## Phase 3: distilled student

```bash
torchrun --standalone --nproc_per_node=8 \
  legged_lab/scripts/train_collaboration.py \
  --phase=3 \
  --distributed --headless --num_envs=2048 --max_iterations=16000 \
  --no_object_rank_count=2 --right_fixed_rank_count=3 \
  --resume=True --load_run="$PHASE2_RUN" --checkpoint="$PHASE2_CHECKPOINT" \
  --logger=wandb --run_name=phase3_student
```

## Resume within a phase

Use `--resume=True`, `--load_run`, and `--checkpoint` with a checkpoint from
the same phase. Add `--warm_start` to load weights without restoring the
optimizer or iteration.

## Isaac Sim evaluation

```bash
python legged_lab/scripts/evaluate_collaboration.py \
  --task=cola_phase_3_student_left_fixed_bar \
  --load_run=/absolute/path/to/phase3/run \
  --checkpoint=model_XXXX.pt --num_envs=1
```

Add `--headless` to evaluate without a viewer.
