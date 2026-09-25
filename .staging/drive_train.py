# Drive the real Studio training subprocess entrypoint (same config shape as /api/train/start).
import json, os, sys, queue, threading, time, multiprocessing as mp
backend, data, model, expect = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, backend); os.chdir(backend)
from models.training import TrainingStartRequest
from core.training.worker import run_training_process
out = os.path.expanduser(f"~/.unsloth/studio/outputs/staging-11851-{int(time.time())}")
req = TrainingStartRequest(model_name=model, training_type="LoRA/QLoRA", format_type="auto",
    local_datasets=[data], eval_steps=0.5, max_steps=4, batch_size=1, gradient_accumulation_steps=1,
    max_seq_length=512, is_dataset_image=True, load_in_4bit=True, save_steps=0, vision_image_size=256)
cfg = req.model_dump()
cfg.update(dict(hf_token="", allow_ambient=True, optim="adamw_8bit", output_dir=out,
    gradient_checkpointing="unsloth", subject="staging", resolved_gpu_ids=None,
    require_exact_resume_resources=False, require_exact_model_resource=False,
    require_exact_dataset_resource=False, require_validated_model_snapshot=False,
    s3_config=None, model_local_path=None, actual_model_repo_id=None, resume_model_load_mode=None))
eq, sq = mp.Queue(), mp.Queue()
events = []
done = threading.Event()
def reader():
    while not (done.is_set() and eq.empty()):
        try: ev = eq.get(timeout=0.5)
        except queue.Empty: continue
        events.append(ev)
        if ev.get("type") != "status":
            print("EVENT", json.dumps(ev, default=str)[:600], flush=True)
th = threading.Thread(target=reader, daemon=True); th.start()
try:
    run_training_process(event_queue=eq, stop_queue=sq, config=cfg)
except BaseException as e:
    print("RAISED", repr(e), flush=True)
    events.append({"type": "error", "error": repr(e)})
time.sleep(3); done.set(); th.join(10)
errors = [e.get("error", "") for e in events if e.get("type") == "error"]
evals = [e["eval_loss"] for e in events if e.get("type") == "progress" and e.get("eval_loss") is not None]
complete = any(e.get("type") == "complete" for e in events)
split_err = any("train_test_split" in (x or "") for x in errors)
print(f"SUMMARY model={model} complete={complete} eval_losses={evals} split_error={split_err} errors={[x[:200] for x in errors]}")
if expect == "pass":
    ok = complete and len(evals) >= 1 and not errors
elif expect == "split_error":
    ok = split_err
else:
    ok = False
print("VERDICT", "OK" if ok else "UNEXPECTED", "expect=" + expect)
sys.exit(0 if ok else 3 if not (complete or errors) else 1)
