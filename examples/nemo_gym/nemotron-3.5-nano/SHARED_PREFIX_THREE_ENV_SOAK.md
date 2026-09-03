# Shared-prefix three-environment soak plan

This run matrix compares the dense-observe (`off`) and fused shared-prefix
training (`on`) arms on `reasoning_gym`, `citation`, and `freeform`.

## Pinned recipe

- NeMo-RL: `64ea1110c0cf12edd0427f8577c2697bbf13799e` plus the strict-pair
  evidence commits and this recipe commit.
- Megatron Bridge: `166164da6de0ee9bbe2906175b9e7c41411d084b`.
- Megatron Core: `8b8870ad4a54f24a7177f8c70e0403933d3f6dce`.
- Schedule: 20 epochs, five fixture prompts per epoch, 100 optimizer steps,
  one prompt and four generations per step.
- Trainer: TP2, CP2, SP enabled, EP4, ETP1, PP1, MTP5, full activation
  recompute.
- Capacity: 4096 model/train tokens and 16384 train/logprob packing tokens
  in every environment so a complete K=4 cohort fits one shared-prefix bin.
- Logging: W&B online in `nvidia/nano35-rlvr-convergence`; TensorBoard off.
- Scheduler: `batch`, account `nemotron_sw_post`, QoS `normal`.

The strict launcher submits the OFF and ON arms concurrently. Each arm uses
one four-GPU node with colocated policy and generation, matching the geometry
that completed the two-step freeform GPU canary. One environment therefore
uses two nodes/eight GPUs; all three environments use six nodes/24 GPUs.

A split policy/generation alternative would use two nodes/eight GPUs per arm,
or 12 nodes/48 GPUs for the full matrix. It may reduce colocated refit/sleep
overhead, but it is a different geometry and is not the default comparison.

## Release gates

Do not submit the six 100-step arms until all of the following are true:

1. The PR-consistent NeMo/Bridge/MCore stack has one immutable deployment and
   externally recorded READY and runnable-manifest digests.
2. The strict captured-output replay producer, terminal evidence seal, and
   isolated evaluator are frozen and pass their full tests.
3. Exact-container config composition passes for all three YAMLs.
4. A short all-five-row live calibration demonstrates nonconstant reward for
   Reasoning Gym and citation; freeform already demonstrated it in the
   two-step GPU pair.
5. The operator explicitly confirms the six-node/24-GPU launch.

`launch_pair.sh --dry-run` is the final non-executing validation for each
environment. `launch_pair.sh --submit` must be invoked separately with
`STRICT_PAIR_ENVIRONMENT` set to `reasoning_gym`, `citation`, and `freeform`,
and with distinct filesystem-safe `PAIR_ID` values. The launcher owns exact
OFF/ON W&B names, receipts, snapshots, Slurm exports, and parallel submission.
