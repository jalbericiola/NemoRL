# Shared-Prefix Policy Training

This document describes the experimental shared-prefix path for GRPO-style
policy training. It also describes the observation mode that can be used to
measure the opportunity on an existing run without changing model execution.

## Motivation

Policy rollouts commonly produce several completions for the same prompt. A
conventional training batch repeats the prompt once per completion:

```text
[prompt, completion 1]
[prompt, completion 2]
...
[prompt, completion G]
```

Shared-prefix training instead represents a complete rollout group as one star:

```text
[prompt, completion 1, completion 2, ..., completion G]
```

The prompt is evaluated once. Each completion has a causal dependency on the
prompt and its own preceding completion tokens, but never on a sibling
completion. The resulting token-work saving for a group with prompt length
`P` and group size `G` is `P * (G - 1)`.

## Configuration

The feature is controlled under the policy configuration:

```yaml
policy:
  shared_prefix_training:
    mode: disabled  # disabled, observe, or train
```

- `disabled` preserves the existing data path and model execution.
- `observe` preserves model execution and logs exact opportunity metrics.
- `train` enables the experimental shared-prefix packing and Hybrid model path.

`observe` is intended to be safe on existing recipes, including recipes whose
parallel topology is not yet supported by `train`.

## Group Identity and Correctness

Every rollout row is stamped with a stable shared-prefix group ID and its
unpadded prompt length. The group ID is authoritative. Rows with different IDs
are never merged merely because their prompt tokens happen to be equal.

A group is eligible only when all of the following hold:

1. It contains exactly the configured number of generations.
2. Every row has the same complete, unpadded prompt token sequence.
3. The whole group is local to the same data-parallel rank and execution slot.

An incomplete group uses the conventional per-row path. A group ID that maps
to different prompt tokens is treated as corrupted metadata and fails instead
of silently sharing the wrong state.

The driver shards complete groups coherently across data-parallel ranks and
assigns fixed execution slots using conservative padded lengths. Every rank
therefore executes the same number of real forwards; dummy model forwards are
not used. If a star cannot be constructed within its prescribed slot, the
whole slot falls back to conventional execution.

## Model Execution

An eligible group is lowered to a packed star with explicit gather, position,
predecessor, and scatter maps. Position IDs continue from the prompt
independently for every completion.

Hybrid models use two corresponding operations:

- Attention layers use a forest-causal mask. Prompt tokens attend causally
  within the prompt; completion tokens attend to the prompt and to earlier
  tokens in their own branch only.
- Mamba layers scan the prompt once, fork the resulting recurrent state, and
  scan every completion from that state.

The logits remain in packed shape. Next-token log probabilities are gathered
in bounded chunks and fanned back out as scalar per-row values; the
implementation does not materialize a repeated `[G, S, vocabulary]` tensor.
The source row order is restored before returning results to the algorithm.

## Metrics

Observation mode logs the following under `shared_prefix/`:

- `exact_group_sequence_coverage`, `complete_groups`, and
  `fallback_sequences` report whether complete groups are available locally.
- `prompt_tokens`, `shareable_prompt_tokens`, and `prompt_token_fraction`
  measure exact prompt reuse.
- `ideal_shared_token_work`, `ideal_token_reduction`, and
  `ideal_token_work_speedup` describe the exact token-work opportunity after
  fallback accounting.
- `valid_loss_tokens`, `valid_to_total_token_ratio`,
  `non_loss_suffix_tokens`, and `non_loss_suffix_token_fraction` separate
  repeated prompts from other masked tokens.
- `loss_ratio_upper_bound_token_reduction` is the optimistic bound obtainable
  from valid/total token counts alone.

For valid loss tokens `V`, total unpadded row tokens `T`, and generation count
`G`, the valid/total ratio gives this optimistic token-work bound:

```text
upper-bound reduction = (1 - V/T) * (G - 1) / G
upper-bound speedup   = 1 / (V/T + (1 - V/T) / G)
```

This calculation assumes that every non-loss token is a repeated prompt token
in a complete group. In practice, `T - V` can also contain masked suffix,
intermediate environment, invalid, or truncated tokens. The ratio therefore
cannot determine exact prompt lengths, group completeness, rank locality, or
realized wall-clock speed. Use the exact metrics above to distinguish the
shareable prompt from those effects.

## Supported Training Slice

`train` is deliberately fail-closed. The current capability is limited to:

- Megatron Hybrid models with the matching shared-prefix capability marker.
- tensor parallel, context parallel, and pipeline parallel sizes all equal to
  one; sequence parallelism disabled.
- sequence packing enabled and MTP disabled.
- no full/core-attention recompute, fine-grained activation offload, training
  CUDA graphs, FP8, or FP4.
- zero attention and hidden dropout, standard RoPE, vanilla softmax, no
  sliding-window attention, and no multi-latent attention.
- deterministic MoE routing without auxiliary or z losses, router jitter,
  capacity dropping, forced/random routing, or expert-bias updates.
- no QK clipping or log-max-attention-logit modification.
- sampling without top-k or top-p log-probability truncation.

Unsupported resolved model settings raise before the first training forward.
This is important because a configuration field that is dormant for a dense
model does not necessarily make that dense model unsupported.

The fused forest-attention implementation retains a Triton merge kernel for
future work, but production use is hard-disabled pending a fix for observed
full-model corruption; the old environment opt-in fails closed. Right-padded
completion branches also mean that ideal token work is an opportunity measure,
not a promise of proportional kernel or wall-clock speedup.

## Validation

Before enabling `train` for a new model or topology, compare conventional and
shared-prefix execution on the same batch. Validate forward loss and per-token
log probabilities, input gradients, parameter gradients, source-row ordering,
and optimizer updates. Hybrid validation must cover both attention and Mamba
layers, and MoE models must exercise an expert-containing layer pattern rather
than only dormant MoE configuration fields.
