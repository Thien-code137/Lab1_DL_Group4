"""TV4 — chỉ tạo hình từ dữ liệu thực tế."""

def plot_history(history_csv, output_path):
    """Vẽ train_loss/val_loss theo epoch, nhãn trục và legend, lưu PNG rồi close.
    Return str đường dẫn PNG.
    """
    raise NotImplementedError("TV4: triển khai plot_history")

def plot_predictions(model, dataset, indices, class_names, device, output_path):
    """Mỗi ảnh hiển thị Actual/Predicted; eval/no_grad; đúng cặp ảnh và nhãn.
    Return str đường dẫn PNG. Nếu chỉ vẽ lỗi phải chọn từ lỗi thực tế.
    """
    raise NotImplementedError("TV4: triển khai plot_predictions")
