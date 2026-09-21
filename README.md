# G1 Fall Recovery

A Unitree G1 humanoid learning to stand back up from the ground, with reinforcement
learning in Isaac Lab.

The policy starts from a state a real fall produces (lying prone, supine or on its
side) and has ten simulated seconds to get its pelvis back to standing height and hold
it there.

<p align="center">
  <img src="media/07-final-policy.gif" width="520" alt="The trained policy standing four G1 robots up from the ground, with no assistance">
</p>

<p align="center"><em>Final policy, deterministic inference, no assist force.</em></p>

## Result

| Metric | Value |
|---|---|
| `stand_up_exp` (max 2.0) | 1.95 |
| Pelvis height error vs. the 0.74 m target | about 8 cm, averaged over the episode |
| Base tilt | `projected_gravity_b[:, 2]` ≈ −0.99 (vertical) |
| Horizontal drift | about 0.27 m/s RMS, so it stands up in place |
| Training cost | 3000 PPO iterations, ~295M simulated steps, ~40 min on one RTX 4090 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/00-curves-dark.png">
  <img src="media/00-curves-light.png" alt="Episode_Reward/stand_up_exp across three runs: flat at 0.31 without the assist, rising to about 1.96 with it, and holding at 1.95 once the posture penalties are switched on">
</picture>

The blue curve is the run without the assist force. With the same reward, the same
initial states and 3000 iterations of PPO, the robot never learns to stand up. The
assist curriculum is what unlocks the task, not the reward weights.

## Setup

| | |
|---|---|
| Simulator | Isaac Lab 2.3.2 / Isaac Sim 5.1 |
| Robot | `G1_MINIMAL_CFG`, 44 bodies, 37 DOF |
| Parallel envs | 4096 |
| Episode | 10 s, 500 steps at 50 Hz control, 200 Hz physics |
| Observations | 106: base linear and angular velocity, projected gravity, joint positions and velocities, last action |
| Actions | 23 joint position targets, the 14 finger joints excluded |
| Algorithm | PPO (rsl_rl), actor and critic MLP `[512, 256, 128]`, ELU |
| Terrain | flat plane |

Only `time_out` ends an episode. There is no early termination on purpose: on a get-up
task, a contact-based termination fires on the state the robot is supposed to recover
from.

### Reward

One term carries the objective:

```python
stand_up_exp = exp(-(h - 0.74)² / 0.5² - (g_z + 1)² / 1.0²)      # weight 2.0
```

`h` is the pelvis height, `g_z` the vertical component of gravity in the base frame
(−1 upright, 0 on the side, +1 upside down).

The two halves are multiplied rather than added, which matters (see stage 1).

The rest is shaping and regularisation: `track_height` and `upright_exp` at weight 0.25
so both halves stay readable in the logs, plus posture penalties introduced by
curriculum.

## How it got there

### 1. Additive rewards, and reward hacking

<img src="media/01-reward-hacking.gif" width="440" alt="Robots sitting upright on the ground with their legs splayed, never lifting themselves">

Height and uprightness were two separate reward terms, added together. The policy
converged to `upright_exp` = 0.96 out of 1.0 while `track_height` stayed at 0.12. It had
found a pose that collects the uprightness reward without lifting its own weight:
sitting on the floor, legs splayed, torso vertical.

That is 44% of the available task reward for no effort, and the saturated term gave no
gradient to do better.

Fix: multiply the two kernels instead of adding them. Upright while flat on the floor
then scores as poorly as lying down at the right height.

### 2. Multiplicative reward, and the exploration wall

<img src="media/02-flailing.gif" width="440" alt="Robots thrashing on the ground without managing to stand">

The exploit was gone and the robot learned nothing. Between iteration 930 and 2057, so
110M simulated steps, `stand_up_exp` moved by 0.0006. Standing up is a coordinated two
to three second motion, and Gaussian noise on 23 joints does not find it.

### 3. A bank of real fallen states

Spawning the robot in mid-air with a random orientation taught it to catch itself when
dropped, which is not the same task.

`scripts/make_fallen_bank.py` drops 512 robots from 0.9 m with random orientations and
random held joint targets, lets physics settle them for five seconds, and records the
resulting root pose and joint configuration. Eight rounds give 4096 states, sampled at
every reset.

### 4. The upward assist curriculum

Taken from HoST: hold the robot up early so it can reach and be rewarded for standing
states, then remove the help.

An external force is applied to `torso_link` in the world frame (the robot starts lying
down, so "up" cannot be the body frame), worth 80% of body weight, annealed linearly to
zero over the first 30 000 environment steps, around iteration 1250.

That change took `stand_up_exp` from 0.31 to 1.96. The policy stands unaided for the
last 1750 iterations, after the assist is gone.

### 5. Posture penalties, introduced by curriculum

The robot stood up, but twitched and wandered off. Penalties on horizontal velocity,
foot slide, torso rotation and action rate fixed that, but only once they were
introduced after the get-up was already learned.

Applied from step zero they collapsed the run: staying still became cheaper than
moving, the action noise std fell from 0.85 to 0.13, entropy went negative, and the
policy settled back onto the floor.

`CurriculumCfg` switches each penalty on at 40 000 steps via `modify_reward_weight`, and
`entropy_coef` went from 0.008 to 0.015. Final `stand_up_exp` is 1.95 instead of 1.96.

## Training progression

Clips from the final run, evenly spaced. These show training behaviour, so the actions
carry exploration noise. The deployed policy is smoother.

| Step 0, assist at full strength | Step 20 000, assist fading |
|---|---|
| <img src="media/03-train-step0.gif" width="380" alt="Start of training, robots held up by the assist force"> | <img src="media/04-train-step20k.gif" width="380" alt="Mid training, the assist fades and the robots begin to support themselves"> |

| Step 40 000, unaided, penalties switching on | Step 70 000, converged |
|---|---|
| <img src="media/05-train-step40k.gif" width="380" alt="Robots standing without assistance as the posture penalties activate"> | <img src="media/06-train-step70k.gif" width="380" alt="End of training, robots standing up cleanly and holding still"> |

## Reproducing

```bash
# with a Python interpreter that has Isaac Lab installed
python -m pip install -e source/g1_fall_recovery

# 1. record the fallen-state bank (a few minutes)
python scripts/make_fallen_bank.py --headless

# 2. train (~40 min on an RTX 4090)
python scripts/rsl_rl/train.py --task Template-G1-Fall-Recovery-v0 --headless \
  --video --video_interval 10000 --video_length 500

# 3. play the result and export the policy to TorchScript / ONNX
python scripts/rsl_rl/play.py --task Template-G1-Fall-Recovery-v0 --headless \
  --num_envs 4 --video --video_length 500
```

A pre-recorded bank is committed at
`source/g1_fall_recovery/g1_fall_recovery/tasks/manager_based/g1_fall_recovery/mdp/fallen_states.pt`,
so step 1 is optional.

The GIFs and checkpoints are stored with Git LFS. Run `git lfs install` before cloning,
or they arrive as text pointer files.

## Layout

```
scripts/
  make_fallen_bank.py            # records the bank of settled fallen states
  rsl_rl/train.py, play.py
source/g1_fall_recovery/g1_fall_recovery/tasks/manager_based/g1_fall_recovery/
  g1_fall_recovery_env_cfg.py    # scene, observations, rewards, events, curriculum
  mdp/rewards.py                 # stand_up_exp, track_height, upright_exp, base_lin_vel_xy_l2
  mdp/events.py                  # reset_from_fallen_bank, apply_upward_assist
  agents/rsl_rl_ppo_cfg.py
```

## Next

- Rough terrain. The flat plane was chosen to separate getting up from handling the ground.
- Sim-to-sim validation in MuJoCo before anything touches hardware.
- A wider bank. The current one is built from free-fall drops only, not from falls that
  follow a push during locomotion.
