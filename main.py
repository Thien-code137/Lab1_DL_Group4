"""TV6 — điểm chạy nền; --mode check không cần torch/torchvision."""
import argparse
import ast
from pathlib import Path
from src.utils import ROOT, project_path, load_json, save_json, validate_config


def check_foundation():
    required = [
        "main.py", "README.md", "requirements.txt", ".gitignore",
        "docs/TEAM_CONTRACT.md", "docs/FILE_OWNERS.md", "docs/REVIEW_STATUS.md",
        "docs/LAB1_Phan_cong_nhom_6_thanh_vien.md",
        "data/README.md", "outputs/README.md", "reports/Lab1_Report.md",
        "notebooks/01_data_check.ipynb",
        *[f"configs/{run}.json" for run in "ABCD"],
        *[f"src/{name}.py" for name in ("__init__", "data", "model", "train", "evaluate", "visualize", "experiments", "predict", "utils")],
        *[f"reports/sections/{name}.md" for name in ("01_data", "02_model", "03_training", "04_evaluation", "05_experiments", "06_integration")],
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu file cấu trúc: {missing}")
    configs = {}
    for run in "ABCD":
        config = validate_config(load_json(ROOT / f"configs/{run}.json"))
        if config["run_id"] != run:
            raise ValueError(f"run_id không khớp {run}.json")
        configs[run] = config
        print(f"OK config: {run}.json")
    fixed = ("data_root", "split_indices_path", "train_size", "val_size", "seed",
             "batch_size", "num_workers", "transform", "model_name", "optimizer", "epochs", "device", "output_root")
    for run in "BCD":
        if any(configs[run][key] != configs["A"][key] for key in fixed):
            raise ValueError(f"Run {run} thay đổi yếu tố cần giữ cố định")
    expected = {"A": (128, .001), "B": (128, .0001), "C": (128, .01), "D": (256, .001)}
    for run, pair in expected.items():
        if (configs[run]["hidden_units"], configs[run]["learning_rate"]) != pair:
            raise ValueError(f"Run {run} lệch kế hoạch khảo sát đã chốt")
    signatures = {
        "data": {"prepare_data": ["config"], "build_dataloaders": ["config"]},
        "train": {"train_one_epoch": ["model", "loader", "loss_fn", "optimizer", "device"],
                  "fit": ["model", "train_loader", "val_loader", "config", "run_dir"]},
        "evaluate": {"evaluate": ["model", "loader", "loss_fn", "device"],
                     "collect_predictions": ["model", "loader", "device"],
                     "evaluate_final": ["checkpoint_path", "device"]},
        "visualize": {"plot_history": ["history_csv", "output_path"],
                      "plot_predictions": ["model", "dataset", "indices", "class_names", "device", "output_path"]},
        "experiments": {"summarize_and_select": ["run_dirs", "output_root"]},
        "predict": {"load_model": ["checkpoint_path", "device"],
                    "predict_batch": ["model", "images", "device"],
                    "check_reload": ["model", "images", "checkpoint_path", "device", "output_path"]},
    }
    todos = []
    for path in [ROOT / "main.py", *sorted((ROOT / "src").glob("*.py"))]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, args in signatures.get(path.stem, {}).items():
            if name not in functions or [a.arg for a in functions[name].args.args] != args:
                raise ValueError(f"Sai giao diện {path.name}:{name}")
        if path.stem == "model":
            cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLP"), None)
            methods = {} if cls is None else {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            for name, args in {"__init__": ["self", "hidden_units"], "forward": ["self", "images"]}.items():
                if name not in methods or [a.arg for a in methods[name].args.args] != args:
                    raise ValueError(f"Sai giao diện MLP.{name}")
        for n in ast.walk(tree):
            if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "NotImplementedError":
                todos.append(f"{path.relative_to(ROOT)}:{n.lineno}")
    notebook = load_json(ROOT / "notebooks/01_data_check.ipynb")
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Notebook khung không hợp lệ")
    print(f"OK {len(required)} file bắt buộc; cú pháp, giao diện, notebook và cấu hình khảo sát.")
    print(f"Còn {len(todos)} vị trí NotImplementedError thuộc phần thuật toán của thành viên.")
    for item in todos:
        print(f"  TODO {item}")
    print("CHƯA xác nhận tải data, train, metrics hoặc reload; check chỉ kiểm cấu trúc/giao diện.")


def main():
    parser = argparse.ArgumentParser(description="FashionMNIST Lab 1 — bộ khung nhóm")
    parser.add_argument("--mode", choices=["check", "prepare", "train", "select", "evaluate"], required=True)
    parser.add_argument("--config", default="configs/A.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--runs", nargs="+", default=["A", "B", "C", "D"])
    args = parser.parse_args()
    if args.mode == "check":
        check_foundation()
        return
    from src.utils import set_seed, resolve_device, collect_environment
    if args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("evaluate cần --checkpoint")
        checkpoint = project_path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Chưa có checkpoint: {checkpoint}")
        from src.evaluate import evaluate_final
        print(evaluate_final(str(checkpoint), resolve_device("auto")))
        return
    config = validate_config(load_json(project_path(args.config)))
    if args.mode == "select":
        from shutil import copy2
        from src.experiments import summarize_and_select
        output_root = project_path(config["output_root"])
        final_path = output_root / "final" / "best_model.pth"
        if final_path.exists():
            raise FileExistsError("Checkpoint cuối đã tồn tại; lưu bản cũ và xác nhận trước khi chọn lại")
        run_dirs = []
        for run in args.runs:
            validate_config({**config, "run_id": run})
            run_dir = output_root / "runs" / run
            for name in ("config.json", "metadata.json", "history.csv", "best_model.pth"):
                if not (run_dir / name).is_file():
                    raise FileNotFoundError(f"Run chưa đủ output: {run_dir / name}")
            run_dirs.append(run_dir)
        result = summarize_and_select(run_dirs, output_root)
        source = project_path(result["checkpoint_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Không có checkpoint được chọn: {source}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, final_path)
        print(result)
        print(f"Checkpoint cuối: {final_path}")
        return
    if args.mode == "prepare":
        from src.data import prepare_data
        print(prepare_data(config))
        return
    split_path = project_path(config["split_indices_path"])
    if not split_path.is_file():
        raise FileNotFoundError("Chưa có split dùng chung: TV1 cần hoàn thành và chạy mode prepare trước")
    run_dir = project_path(config["output_root"]) / "runs" / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Run đã tồn tại: {run_dir}. Đổi run_id trước khi chạy lại.")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    from src.data import build_dataloaders
    from src.model import MLP
    from src.train import fit
    train_loader, val_loader, _test_loader, class_names = build_dataloaders(config)
    model = MLP(hidden_units=config["hidden_units"]).to(device)
    # Metadata phục vụ checkpoint; không lưu object Python tùy biến.
    config = {**config, "resolved_device": str(device), "class_names": list(class_names)}
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    metadata = collect_environment()
    metadata.update({"run_id": config["run_id"], "seed": config["seed"], "device": str(device)})
    save_json(run_dir / "metadata.json", metadata)
    result = fit(model, train_loader, val_loader, config, run_dir)
    from src.visualize import plot_history
    plot_history(result["history_path"], run_dir / "loss_curves.png")
    print(result)


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, FileNotFoundError, FileExistsError, ValueError, ModuleNotFoundError) as exc:
        raise SystemExit(f"Chưa thể hoàn thành tác vụ: {exc}")
