"""TV5 — dùng kết quả train chung, không triển khai training loop thứ hai."""

def summarize_and_select(run_dirs, output_root):
    """Đọc config/history/metadata từng run; xuất experiment_results.csv.
    Chọn run bằng best_val_accuracy, hòa bằng best_val_loss; nếu vẫn hòa giữ
    run xuất hiện trước trong run_dirs. Ghi final/selection.json (run_id,
    best_epoch, checkpoint_path, val_metrics, criterion).
    Không dùng test. Return dict {results_path, selection_path, checkpoint_path}.
    Không điền dữ liệu giả khi run thất bại/chưa có checkpoint.
    """
    raise NotImplementedError("TV5: triển khai tổng hợp và chọn run")
