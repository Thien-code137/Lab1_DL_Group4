"""TV6 — nền lưu/tải; kiểm tra thực tế sau khi TV2/TV3 hoàn thành."""
from pathlib import Path
from .utils import project_path

def load_model(checkpoint_path, device):
    import torch
    from .model import MLP
    path = project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có checkpoint: {path}")
    # Checkpoint của nhóm chỉ gồm state_dict và metadata kiểu cơ bản.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"model_state_dict", "model_config", "preprocessing", "class_names",
                "run_id", "epoch", "val_metrics"}
    if not isinstance(checkpoint, dict) or required - checkpoint.keys():
        raise ValueError("Checkpoint thiếu khóa theo TEAM_CONTRACT")
    model_config = checkpoint["model_config"]
    if model_config.get("model_name") != "mlp":
        raise ValueError("Bộ khung chỉ hỗ trợ model_name=mlp")
    if checkpoint["preprocessing"].get("transform") != "to_tensor":
        raise ValueError("Transform chưa được triển khai trong contract nền")
    if len(checkpoint["class_names"]) != 10:
        raise ValueError("Checkpoint cần đúng 10 tên lớp")
    model = MLP(hidden_units=model_config["hidden_units"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return model, metadata

def predict_batch(model, images, device):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        return {"logits": logits.cpu(), "labels": logits.argmax(dim=1).cpu()}


def check_reload(model, images, checkpoint_path, device, output_path):
    """TV6: so model hiện tại với đúng checkpoint của nó, không dùng model last epoch.
    Gọi ngay sau khi lưu best hoặc khi cả hai model cùng nạp đúng best checkpoint.
    Lần đầu kiểm tra nên làm trong smoke integration cùng một model trước/sau save.
    """
    import torch
    from .utils import save_json
    before = predict_batch(model, images, device)
    loaded, metadata = load_model(checkpoint_path, device)
    after = predict_batch(loaded, images, device)
    logits_match = torch.allclose(before["logits"], after["logits"], rtol=1e-5, atol=1e-6)
    labels_match = torch.equal(before["labels"], after["labels"])
    result = {"logits_match": bool(logits_match), "labels_match": bool(labels_match),
              "rtol": 1e-5, "atol": 1e-6, "num_samples": int(images.shape[0]),
              "run_id": metadata["run_id"], "epoch": metadata["epoch"],
              "checkpoint_path": str(project_path(checkpoint_path)),
              "passed": bool(logits_match and labels_match)}
    save_json(output_path, result)
    if not result["passed"]:
        raise ValueError("Reload check thất bại; xem JSON và kiểm tra model/checkpoint nguồn")
    return result
