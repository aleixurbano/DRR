# `reflect.sim`

Simulation pipeline for AI2-THOR based task validation.

## What Lives Here

- `actions.py` - Action primitive execution (pick up, put, toggle, etc.)
- `task_manager.py` - Task state utilities, failure handling, navigation helpers
- `recovery.py` - Replan execution (normalize and run typed/legacy plans)
- `data_gen.py` - Synthetic episode generation with failure injection
- `assets/sounds/` - Sidecar WAV files for audio-augmented perception

## Output Contract

Simulation summaries and correction artifacts are routed under:

```text
$REFLECT_OUTPUT_ROOT/sim_validation/
```

Use the CLI entry point rather than calling these modules directly:

```bash
reflect-sim --tasks boilWater --episodes boilWater-1 --backend local --model qwen3.5:9b
```
