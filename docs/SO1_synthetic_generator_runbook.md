# SO1 synthetic generator runbook

This generator implements the SO-1 synthetic data construction path on top of
the frozen SO-0 v1.1 camera renderer. It does not modify SO-0 assets, formulas,
thresholds, or frozen reports.

## Local validation

From the project root:

```bash
PYTHONPATH=src:src/skin_optics_so0/src python -m skin_optics_so1.build_synthetic_dataset \
  --config configs/so1_synthetic_generation_v1.yaml \
  --output-root outputs/SO1_Synthetic_Generator_v1_validate \
  --validate-only \
  --overwrite-confirmed
```

`--validate-only` checks strict YAML parsing, SO-0 range compatibility,
qualified camera-light pairs, deterministic camera split, manifest legality,
and output partition disk space. It does not render.

## Local smoke

Smoke requires a CUDA-capable PyTorch install. The current WSL environment used
during implementation had CPU-only PyTorch, so GPU smoke was not executed here.

```bash
PYTHONPATH=src:src/skin_optics_so0/src python -m skin_optics_so1.build_synthetic_dataset \
  --config configs/so1_synthetic_smoke_v1.yaml \
  --output-root outputs/SO1_Synthetic_Generator_v1_smoke \
  --smoke-only \
  --device cuda \
  --overwrite-confirmed
```

Smoke renders 64 samples across all six splits and writes small split memmaps,
manifest CSV/JSONL, QC CSV/JSON/Markdown, and preview PNGs.

## A6000 full generation

Run only on the Linux server with CUDA PyTorch and sufficient output space:

```bash
PYTHONPATH=src:src/skin_optics_so0/src python -m skin_optics_so1.build_synthetic_dataset \
  --config configs/so1_synthetic_generation_v1.yaml \
  --output-root /path/to/a6000/data/SO1_Synthetic_Generator_v1 \
  --device cuda
```

Full mode performs preflight, an internal manifest/QC setup, and then writes
per-split `.npy` memmaps:

- `linear_rgb.f16.npy`: `[N,3,256,256]`
- `target_mhsp.f16.npy`: `[N,4,256,256]`
- `valid_mask.u8.npy`: `[N,1,256,256]`
- `written.u8.npy`: `[N]`
- `metadata.csv`

The output path is not hard-coded. Do not run full mode on local WSL; the CLI
rejects default full generation on WSL.

## Resume

Use `--resume` only with the same config, manifest, and output directory. The
storage layer skips samples where `written[index] == 1`. If configuration or
manifest hashes differ, start a new output directory rather than mixing runs.

