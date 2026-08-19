# MuJoCo locomotion sim2sim

The deployment directory contains one sim2sim path: a 29-action policy running
on the fixed-hand G1 without a carried object. It accepts either a Phase-1
actor or a Phase-3 student trained on the no-object population.

The bundled `student.jit` is the exported Phase-3 `model_5000` student and is
loaded by default.

## Export a checkpoint

```bash
python deployment/mujoco/export_policy.py \
  --checkpoint /path/to/model_XXXX.pt \
  --output /path/to/policy.jit
```

The exporter selects the checkpoint's `actor.*` or `student.*` MLP and verifies
its 29-action output before creating the TorchScript file.

## Run

```bash
python deployment/mujoco/run_sim2sim.py \
  --policy /path/to/policy.jit
```

Run the bundled student with:

```bash
python deployment/mujoco/run_sim2sim.py
```

For a Phase-1 actor, the four command values are forward velocity, lateral
velocity, yaw velocity, and body height. Inputs are rejected outside their
training ranges:

```bash
python deployment/mujoco/run_sim2sim.py \
  --policy /path/to/phase1_policy.jit \
  --command 0.3 0.0 0.0 0.78
```

- forward velocity: `[-0.75, 1.05] m/s`;
- lateral velocity: `[-0.50, 0.50] m/s`;
- yaw velocity: `[-1.20, 1.20] rad/s`;
- height: `[0.45, 0.90] m`.

A Phase-3 student uses its exact no-object training observation contract:
25 frames, zero masked planar command, and masked height `0.78 m`. Passing a
different `--command` is rejected rather than silently creating an
out-of-distribution observation.

Run a finite headless rollout or the built-in in-domain schedule with:

```bash
python deployment/mujoco/run_sim2sim.py \
  --headless --duration 20

python deployment/mujoco/run_sim2sim.py \
  --policy /path/to/phase1_policy.jit --headless --schedule
```

## Physics contract

- MuJoCo physics: 1000 Hz;
- policy and position targets: 50 Hz;
- actuator gains and 0.01 armature: the nominal values used by training;
- passive joint damping and friction: inherited from the validated bar demo
  G1 asset;
- solver and ground contact parameters: identical to the validated bar demo;
- observation: 111 values per frame, stacked for 10 frames for Phase 1 or 25
  frames for Phase 3.

The bar body, wrist attachments, external bar controllers, and their assets are
not included in this deployment target.
