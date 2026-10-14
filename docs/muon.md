# Muon optimizer

`src.optim.Muon` (`src/optim/muon.py`) implements Muon (Jordan et al.
2024) with the AdamW-compatible scaling and decoupled weight decay of
"Muon is Scalable for LLM Training" (Liu et al., 2025,
arXiv:2502.16982).

## What it does

For a 2D weight, the SGD-momentum (Nesterov) update is replaced by its
orthogonalization (singular values ~1), computed with 5 quintic
Newton-Schulz iterations in bfloat16. The update is then scaled per
matrix:

- `adjust_lr_fn: match_rms_adamw` (default): `0.2 * sqrt(max(A, B))`,
  so that the update RMS matches AdamW's. **`lr` and `weight_decay` are
  those of the AdamW baseline**, making a fair single-variable
  comparison possible.
- `adjust_lr_fn: original` (Jordan): `sqrt(max(1, A / B))`. Needs a much
  larger Muon `lr` (~0.02) and a smaller AdamW one, set with the model
  hparam `muon_adamw_lr_scale` (AdamW lr = `lr * muon_adamw_lr_scale`).

As recommended since the original Muon, it is only applied to 2D hidden
weights, and the rest is optimized with AdamW. Our `Muon` optimizer is
thus hybrid (unlike `torch.optim.Muon`, which only accepts 2D params):
param groups flagged `use_muon=True` use Muon, the others use AdamW.
`SemanticSegmentationModule.configure_optimizers` builds the groups
automatically when the configured optimizer is a `Muon`
(`split_muon_param_groups`), keeping the existing `transformer_lr_scale`
split. The routing is:

| Muon | AdamW |
|---|---|
| 2D weights: transformer qkv / out_proj / RPE, MLPs, edge MLPs, down-pool | biases, norm weights, `mean_scale`, 3D TorchSparse CNN kernels (EZ-SP), classification heads |

Names containing one of `muon_exclude` (model hparam, default `head`,
`classifier`) are forced to AdamW. The share of the trainable parameters
optimized by Muon is logged at startup: about 98% for SPT, 96% for
SuperCluster and 88% for EZ-SP on KITTI-360.

## Use

Add the `optimizer/muon` config after any experiment:

```bash
python src/train.py experiment=semantic/kitti360_ezsp +optimizer=muon
```

or, in an experiment config:

```yaml
defaults:
  - /experiment/semantic/kitti360_ezsp
  - /optimizer/muon
```

Options, as model hparams: `+model.muon_exclude`, `+model.muon_adamw_lr_scale`
and `+model.muon_split_qkv`.

## Notes / limitations

- Single-device implementation (no DDP sharding of the Newton-Schulz).
- By default (`model.muon_split_qkv: True`) the fused `qkv` weights are
  cut into their q, k, v row blocks (`qk_dim * num_heads`,
  `qk_dim * num_heads`, `dim`, so not always equal) and each block is
  orthogonalized and scaled on its own, instead of coupling them in a
  joint orthogonalization.
- Resuming a run from an AdamW checkpoint into Muon (or vice versa)
  does not work: the optimizer states differ.
- Tests: `tests/test_muon.py`.
