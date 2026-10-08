"""Run the 343M pure SSM on FineWeb and resume after service failures."""

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/ssd/ssm")
DATA = ROOT / "data" / "fineweb_10bt_full"
OUTPUT = DATA / "training_balanced_343m"
PYTHON = "/home/nishantj/miniforge3/envs/ssm/bin/python"


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    if (DATA / "train.bin").stat().st_size != 2 * manifest["train_tokens"]:
        raise RuntimeError("FineWeb training data does not match its manifest")

    OUTPUT.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / '.deps' / 'liger'}:{ROOT / '.deps' / 'triton'}"
    env["TRITON_CACHE_DIR"] = str(ROOT / ".cache" / "triton")
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        PYTHON, str(ROOT / "model" / "pure" / "train_fineweb_full.py"),
        "--data-path", str(DATA / "train.bin"),
        "--validation-path", str(DATA / "validation.bin"),
        "--output-dir", str(OUTPUT),
        "--init", "fresh", "--dim", "1536", "--layers", "8",
        "--passes", "4", "--batch-size", "64", "--seq-len", "512",
        "--log-every", "20", "--validate-every", "1000",
        "--checkpoint-every", "1000",
    ]
    (OUTPUT / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    log_path = OUTPUT / "train.log"
    monitor_path = OUTPUT / "jtop.jsonl"
    with log_path.open("a", buffering=1) as output, monitor_path.open("a", buffering=1) as monitor:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT)
        jetson = None
        try:
            from jtop import jtop
            jetson = jtop()
            jetson.start()
        except Exception as exc:
            monitor.write(json.dumps({"event": "jtop_unavailable", "error": str(exc)}) + "\n")
        while process.poll() is None:
            sample = {"time_unix": time.time(), "training_pid": process.pid}
            if jetson is not None:
                try:
                    sample.update({
                        "ram_used_gib": round(jetson.memory["RAM"]["used"] / 2**20, 3),
                        "gpu_load_percent": jetson.gpu["gpu"]["status"]["load"],
                        "gpu_temp_c": jetson.temperature.get("gpu", {}).get("temp"),
                        "total_power_mw": jetson.power.get("tot", {}).get("power"),
                    })
                except Exception as exc:
                    sample["jtop_error"] = str(exc)
            monitor.write(json.dumps(sample) + "\n")
            time.sleep(60)
        if jetson is not None:
            jetson.close()
        monitor.write(json.dumps({"time_unix": time.time(), "event": "training_exit",
                                  "exit_code": process.returncode}) + "\n")
        if process.returncode:
            raise RuntimeError(f"Training exited with status {process.returncode}; see {log_path}")

    lines = log_path.read_text().splitlines()
    if not lines or '"event": "complete"' not in lines[-1]:
        raise RuntimeError("Training exited without a completion record")
    evaluation = [
        PYTHON, str(ROOT / "model" / "pure" / "evaluate_fineweb.py"),
        "--checkpoint", str(OUTPUT / "latest_checkpoint.pt"),
        "--validation-path", str(DATA / "validation.bin"),
        "--output", str(OUTPUT / "evaluation.json"),
    ]
    with (OUTPUT / "evaluation.log").open("a", buffering=1) as evaluation_log:
        result = subprocess.run(evaluation, cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=evaluation_log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Final evaluation exited with status {result.returncode}")


if __name__ == "__main__":
    main()
