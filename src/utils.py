"""TV6 — tiện ích nền; không import torch ở cấp module."""
from pathlib import Path
import json
import random
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[1]

def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config phải là JSON object")
    required = {"run_id", "data_root", "split_indices_path", "train_size", "val_size",
                "seed", "batch_size", "num_workers", "transform", "model_name",
                "hidden_units", "optimizer", "learning_rate", "epochs", "device", "output_root"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Thiếu config: {sorted(missing)}")
    run_id = config["run_id"]
    if not isinstance(run_id, str) or not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
        raise ValueError("run_id chỉ dùng chữ, số, dấu _ hoặc -")
    for key in ("train_size", "val_size", "batch_size", "hidden_units", "epochs"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} phải là số nguyên dương")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} phải là số nguyên không âm")
    import math
    lr = config["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate phải là số dương hữu hạn")
    if config["train_size"] + config["val_size"] != 60000:
        raise ValueError("train_size + val_size phải bằng 60000")
    for key in ("data_root", "split_indices_path", "output_root"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} phải là đường dẫn không rỗng")
    for key, expected in (("transform", "to_tensor"), ("model_name", "mlp"), ("optimizer", "adam")):
        if config[key] != expected:
            raise ValueError(f"Contract nền yêu cầu {key}={expected}")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device phải là auto, cpu hoặc cuda")
    return config

def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Không cam kết bitwise-identical giữa mọi phần cứng/phiên bản.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def resolve_device(name="auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("Đã chọn CUDA nhưng môi trường chưa có CUDA khả dụng")
    return torch.device(name)

def collect_environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "commit_sha": commit}
