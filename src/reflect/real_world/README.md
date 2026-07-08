# `reflect.real_world`

Real-world REFLECT extension pipeline.

## What Lives Here

- `scene_graph.py`, `local_graph.py` - Real-world graph construction and scene reasoning
- `batch_validation.py`, `batch_reasoning.py` - Batch validation orchestration
- `detection.py` - MDETR object detection wrapper
- `worker_pool.py`, `gpu_utils.py` - GPU worker management
- `logging_utils.py` - Logging and profiling
- `diagrams/` - Authored error pipeline diagrams

## Runtime Model

The CLI entry point prepares a runtime workspace under:

```text
$REFLECT_OUTPUT_ROOT/real_world/runtime/
```

That workspace mirrors the old path assumptions without writing back into the repo.

Use:

```bash
reflect-real --tasks 1 2
```
