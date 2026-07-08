# putAppleBowl1 Error Diagram

This diagram is meant to explain two things at once:

1. how the real-world pipeline flows from image -> detector -> filter -> scene graph -> summary -> LLM
2. where the `putAppleBowl1` failure enters the system at frame `1276` (`00:42`)

## Files used

Paths are relative to the real-world runtime workspace
(`$REFLECT_OUTPUT_ROOT/real_world/runtime/`), populated by a `reflect-real` run:

- `real_world/state_summary/putAppleBowl1/mdetr_obj_det/images/1276.png`
- `real_world/state_summary/putAppleBowl1/mdetr_obj_det/det/1276.png`
- `real_world/state_summary/putAppleBowl1/mdetr_obj_det/clip_processed_det/1276.png`
- `real_world/state_summary/putAppleBowl1/state_summary_L1.txt`
- `real_world/state_summary/putAppleBowl1/state_summary_L2.txt`
- `real_world/state_summary/putAppleBowl1/llm_trace.json`

## Regenerate

```bash
python src/reflect/real_world/diagrams/build_put_apple_bowl1_error_diagram.py
```

## Main teaching point

The pipeline is not "hallucinating" out of nowhere. It makes an early visual grounding mistake:

- raw apple boxes are oversized and ambiguous
- CLIP-based confirmation drops the apple
- the bowl is the only surviving object
- the gripper heuristic assigns that bowl to the gripper
- the LLM receives a wrong but internally consistent text summary
