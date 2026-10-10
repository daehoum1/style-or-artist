"""Run style evaluation + probes for every (dataset, encoder) as soon as its embeddings and splits exist.
Exits when extraction has finished (ALLDONE marker) and nothing is left. Usage: run_eval_loop.py <dataset>"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
ds = sys.argv[1]
encs = ["clip_b16", "dinov2_b14", "mae_b16", "siglip_b16", "csd_l14", "vit_b16_in21k", "openclip_b16",
        "dinov3_b16", "clip_l14", "dinov2_l14", "siglip2_l16", "siglip_so400m", "aladin_vit"]
emb_ds = ds
if len(sys.argv) > 2 and sys.argv[2] == "--reverse":
    encs = encs[::-1]
while True:
    pending = 0
    for e in encs:
        emb = A / "embeddings" / emb_ds / e / "embeddings.npy"
        splits_ready = (A / "splits" / ds / "split_summary.csv").exists()
        res = A / "results" / "style_eval" / ds / f"{e}.parquet"
        pr = A / "results" / "probes" / ds / f"{e}.parquet"
        if res.exists() and pr.exists():
            continue
        pending += 1
        if not (emb.exists() and splits_ready):
            continue
        lock = A / "logs" / f"lock_{ds}_{e}"
        busy = subprocess.run(["pgrep", "-f", f"(06_style_eval|07_probes).py {ds} {e}$"], capture_output=True).stdout.strip()
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)   # atomic: exactly one worker gets an encoder
            os.close(fd)
        except FileExistsError:
            continue
        if busy:          # a pre-lock worker is still running this encoder
            continue
        log = open(A / "logs" / f"eval_{ds}_{e}.log", "a")
        if not res.exists():
            subprocess.run([sys.executable, "scripts/06_style_eval.py", ds, e], cwd=ROOT, stdout=log, stderr=log)
        if not pr.exists():
            subprocess.run([sys.executable, "scripts/07_probes.py", ds, e], cwd=ROOT, stdout=log, stderr=log)
        print(f"{time.strftime('%H:%M')} done {ds}/{e}", flush=True)
        if not (res.exists() and pr.exists()):
            lock.unlink()  # failed: let a later pass retry
    done_marker = (A / "logs" / "extract_failures.log").exists() and "ALLDONE" in (A / "logs" / "extract_failures.log").read_text()
    if pending == 0 or (done_marker and all(
            not (A / "embeddings" / emb_ds / e / "embeddings.npy").exists() or
            ((A / "results" / "style_eval" / ds / f"{e}.parquet").exists() and (A / "results" / "probes" / ds / f"{e}.parquet").exists())
            for e in encs)):
        print("loop finished", flush=True)
        break
    time.sleep(60)
