# File nào do ai làm?

| Người | File tự triển khai | Phần nền đã có để dùng | Bàn giao đầu tiên |
|---|---|---|---|
| TV1 | src/data.py, notebooks/01_data_check.ipynb, reports/sections/01_data.md | Config A, utils JSON/project_path; notebook ghi chú mở đầu | Loader và split đúng shape |
| TV2 | src/model.py, reports/sections/02_model.md | Chữ ký MLP/forward | Forward chạy được `[B,10]` |
| TV3 | src/train.py, reports/sections/03_training.md | main tạo model/device/run_dir; contract checkpoint | Một batch backward/step và history |
| TV4 | src/evaluate.py, src/visualize.py, reports/sections/04_evaluation.md | Chữ ký hàm, contract metrics | evaluate trước, hình/test cuối sau |
| TV5 | src/experiments.py, configs/*.json, reports/sections/05_experiments.md | Bốn JSON hợp lệ, cùng baseline | Xác nhận config sớm, rồi tổng hợp/chọn run |
| TV6 | main.py, src/utils.py, src/predict.py, README, docs, requirements, báo cáo cuối | CLI, config validation, seed/device, load_model/predict_batch đã viết nền | Tiếp nhận, cài môi trường, ghép một batch và kiểm tra reload |

Mọi TODO nêu tên người chịu trách nhiệm. Không điền metrics vào outputs bằng tay. TV6 không viết thay thuật toán TV1–5. Mỗi người viết phần báo cáo mình và review phần người nhận bàn giao.

## Chủ sở hữu kết quả tự sinh

- TV1: split_indices.json, data_summary.json, class_distribution.csv, sample_images.png.
- TV3: cơ chế tạo history/checkpoint/metadata; chạy A. TV5 dùng lại cơ chế để chạy B/C/D.
- TV4: loss_curves mỗi run, test_metrics/test_predictions và ảnh cuối.
- TV5: experiment_results.csv và final/selection.json.
- TV6: bản sao checkpoint cuối, reload_check.json và báo cáo tổng hợp.

Xem LAB1_Phan_cong_nhom_6_thanh_vien.md để đọc nhiệm vụ chi tiết và cây thư mục. TEAM_CONTRACT.md là nguồn ưu tiên khi cần xác nhận chữ ký hàm/định dạng tích hợp.
