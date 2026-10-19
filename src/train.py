"""TV3 — training; dùng evaluate của TV4, không viết bản evaluate riêng."""

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """model.train; zero_grad/forward/loss/backward/step.
    Return {loss: float, accuracy: float}; accuracy 0–1.
    Loss epoch = tổng(batch_loss * số mẫu batch) / tổng số mẫu.
    """
    raise NotImplementedError("TV3: triển khai train_one_epoch")

def fit(model, train_loader, val_loader, config, run_dir):
    """Tạo CrossEntropyLoss/Adam một lần, đặt model.train trong mỗi epoch.
    Main đã chọn device, tạo model, set seed và tạo run_dir/config/metadata.
    Fit ghi history.csv, cập nhật metadata thời gian và lưu best_model.pth.
    Dùng evaluate từ src.evaluate. Chọn best theo val_accuracy, hòa theo val_loss.
    Checkpoint: xem TEAM_CONTRACT, preprocessing/model_config KHÔNG được thiếu.
    Epoch bắt đầu 1; metric là float; val_accuracy thang 0–1.
    Return {history_path: str, best_checkpoint_path: str, best_epoch: int}.
    Không dùng test_loader trong fit.
    """
    raise NotImplementedError("TV3: triển khai fit")
