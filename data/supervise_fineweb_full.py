"""Wait for full FineWeb preparation, then run and monitor pure SSM training.

Designed for a user systemd service. A nonzero training exit lets systemd
restart from the last atomic checkpoint.
"""

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/ssd/ssm")
DATA = ROOT / "data" / "fineweb_10bt_full"
PILOT = ROOT / "data" / "fineweb_10bt_fused40m"
PYTHON = "/home/nishantj/miniforge3/envs/ssm/bin/python"


def wait_for_preparation():
    manifest = DATA / "manifest.json"
    config = DATA / "run_config.json"
    while True:
        if manifest.exists() and config.exists():
            return json.loads(config.read_text())
        if not manifest.exists():
            status = subprocess.run(
                ["systemctl", "--user", "is-active", "ssm-fineweb-prep.service"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            if status not in ("active", "activating"):
                raise RuntimeError(f"FineWeb preparation stopped before completion: {status}")
        time.sleep(60)


def main():
    config = wait_for_preparation()
    passes = int(config["passes"])
    if passes not in (1, 4):
        raise ValueError("run_config.json must select one or four passes")

    output_dir = DATA / "training"
    output_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / ".deps" / "liger") + ":" + str(ROOT / ".deps" / "triton")
    env["TRITON_CACHE_DIR"] = str(ROOT / ".cache" / "triton")
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        PYTHON, str(ROOT / "model" / "pure" / "train_fineweb_full.py"),
        "--data-path", str(DATA / "train.bin"),
        "--validation-path", str(DATA / "validation.bin"),
        "--expected-prefix", str(PILOT / "train.bin"),
        "--pilot-checkpoint", str(PILOT / "pure_fused_b32_10min_checkpoint.pt"),
        "--output-dir", str(output_dir),
        "--passes", str(passes),
        "--batch-size", "32", "--seq-len", "512",
        "--log-every", "100", "--validate-every", "5000",
        "--checkpoint-every", "5000",
    ]
    (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    train_log = output_dir / "train.log"
    monitor_log = output_dir / "jtop.jsonl"

    with train_log.open("a", buffering=1) as output, monitor_log.open("a", buffering=1) as monitor:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT)
        jetson = None
        try:
            from jtop import jtop
            jetson = jtop()
            jetson.start()
        except Exception as exc:
            monitor.write(json.dumps({"event": "jtop_unavailable", "error": str(exc)}) + "\n")
            jetson = None

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
            raise RuntimeError(f"Training exited with status {process.returncode}; see {train_log}")
        lines = train_log.read_text().splitlines()
        if not lines or '"event": "complete"' not in lines[-1]:
            raise RuntimeError("Training exited without a completion record")

        evaluation = [
            PYTHON, str(ROOT / "model" / "pure" / "evaluate_fineweb.py"),
            "--checkpoint", str(output_dir / "latest_checkpoint.pt"),
            "--validation-path", str(DATA / "validation.bin"),
            "--output", str(output_dir / "evaluation.json"),
        ]
        with (output_dir / "evaluation.log").open("a", buffering=1) as evaluation_log:
            finished = subprocess.run(evaluation, cwd=ROOT, env=env,
                                      stdin=subprocess.DEVNULL,
                                      stdout=evaluation_log, stderr=subprocess.STDOUT,
                                      check=False)
        if finished.returncode:
            raise RuntimeError(f"Final evaluation exited with status {finished.returncode}")


if __name__ == "__main__":
    main()
