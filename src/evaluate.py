"""TV4 — metrics và test cuối; không import train.py."""

def evaluate(model, loader, loss_fn, device):
    """model.eval + no_grad; không cập nhật trọng số.
    Return {loss: float, accuracy: float}; trung bình theo số mẫu, accuracy 0–1.
    Caller phải gọi lại model.train trước epoch train tiếp theo.
    """
    raise NotImplementedError("TV4: triển khai evaluate sớm cho TV3 sử dụng")

def collect_predictions(model, loader, device):
    """Return {true_labels: list[int], predicted_labels: list[int], indices: list[int]}.
    Contract chỉ hỗ trợ loader không shuffle cho hàm này; indices là thứ tự trong
    dataset loader đang dùng. Với test_dataset gốc: đúng test_index gốc.
    """
    raise NotImplementedError("TV4: triển khai collect_predictions")

def evaluate_final(checkpoint_path, device):
    """Chỉ gọi sau khi TV5 chọn run bằng validation.
    Dùng load_model từ predict, lấy data_root từ metadata/preprocessing.
    Xuất final/test_metrics.json, test_predictions.csv và ảnh Actual/Predicted.
    Return dict đường dẫn output. Không tune dựa trên test.
    """
    raise NotImplementedError("TV4: triển khai evaluate_final sau khi chốt mô hình")
