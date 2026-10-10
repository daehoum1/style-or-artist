"""Environment / version report + source-tree fingerprint -> artifacts/reports/environment.json"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import torch  # noqa: E402

files = sorted([p for d in ["src", "scripts", "configs", "tests"] for p in (ROOT / d).rglob("*")
                if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}])
fp = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files}
tree = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
pkgs = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
keep = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "open_clip", "timm", "transformers", "Pillow",
        "pyarrow", "matplotlib", "PyYAML", "huggingface")
rep = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
       "packages": [p for p in pkgs if p.split("==")[0].startswith(keep)],
       "source_tree_sha": tree, "source_files": fp,
       "git_commit": None, "note": "project is not under a committed git history; source_tree_sha identifies the code"}
(ROOT / "artifacts" / "reports" / "environment.json").write_text(json.dumps(rep, indent=1))
print(tree, rep["packages"])
