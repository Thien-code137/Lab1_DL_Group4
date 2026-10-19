"""TV1 — giữ nguyên giao diện; tham khảo docs/TEAM_CONTRACT.md."""

def prepare_data(config):
    """Tải data, tạo/đọc split, kiểm tra và xuất data_summary/class_distribution/ảnh mẫu.
    Không train. Không ghi đè split đã có; xác minh chỉ số đầy đủ và không giao nhau.
    split JSON: seed, dataset_size, train_indices, val_indices.
    Return: dict các đường dẫn output đã tạo thực tế.
    """
    raise NotImplementedError("TV1: triển khai prepare_data từ 5 bước dữ liệu đã chạy")

def build_dataloaders(config):
    """Dùng split đã chuẩn bị; trả train_loader, val_loader, test_loader, class_names.
    Transform ToTensor; train shuffle với Generator có seed, val/test không shuffle.
    Không tự chia split mới theo từng run. Lấy đường dẫn bằng project_path.
    """
    raise NotImplementedError("TV1: triển khai build_dataloaders")
