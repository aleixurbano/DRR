# `models/`

Local staging area for model weights. Checkpoint files (`*.pt`, `*.pth`, `*.ts`)
are gitignored - nothing heavy is tracked here.

What lands here and where it comes from:

| Weights | Used by | Source |
|---|---|---|
| `yoloe-*-seg.pt` | open-world perception, `analysis/notebooks/perception_yoloe_real.ipynb` | fetched automatically by `ultralytics` on first use |

Other checkpoints (AudioCLIP, MDETR, RAM/Grounded-SAM) are documented in
[docs/data_and_artifacts.md](../docs/data_and_artifacts.md).
