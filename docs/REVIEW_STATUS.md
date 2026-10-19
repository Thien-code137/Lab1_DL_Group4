# Double-check bộ khung — 19/10/2026

## Kết luận

Cấu trúc đủ bao phủ các nhóm yêu cầu Lab 1 và đủ giao phần triển khai cho 6 người. Đây chưa là bài lab chạy hoàn chỉnh: các TODO là phần công việc được giao, không phải kết quả đã thực hiện.

## Điểm tìm thấy và đã sửa

| Vấn đề bản đầu | Điều chỉnh |
|---|---|
| Chưa có lệnh chọn run nối với checkpoint cuối | Thêm mode select, gọi TV5 và chép đúng checkpoint nguồn |
| Check mới kiểm một phần tài liệu/cú pháp | Kiểm tất cả file bắt buộc, cấu hình khảo sát, giao diện hàm/model và notebook |
| Có hàm vẽ nhưng main chưa gọi sau train | Nối plot_history sau fit |
| Kiểm tra reload mới có trong hướng dẫn | Thêm check_reload, xuất JSON khi chạy thật và báo lỗi nếu không khớp |
| Config JSON không phải object gây lỗi khó hiểu | validate_config báo lỗi rõ |
| Notebook và một số module chưa khớp mô tả | Đồng bộ contract, file owners và tài liệu phân công |

## Đối chiếu đề

| Yêu cầu | File chịu trách nhiệm | Tình trạng |
|---|---|---|
| FashionMNIST/transforms/loaders | src/data.py; TV1 | Giao diện có; TODO thuật toán |
| Neural network | src/model.py; TV2 | Lớp có; TODO thuật toán |
| Training/autograd/optimizer | src/train.py; TV3 | Giao diện có; TODO thuật toán |
| Accuracy | src/evaluate.py; TV4 | Giao diện có; TODO thuật toán |
| Save/load | train.py/predict.py; TV3/TV6 | Save TODO; load/check_reload có nền, chưa xác minh bằng torch |
| Experiment | configs + src/experiments.py; TV5 | Config hợp lệ; tổng hợp/chọn TODO |
| Loss graphs | visualize.py và main; TV4 | Đã nối lệnh; hàm vẽ TODO |
| Actual/Predicted | evaluate_final/visualize; TV4 | Hợp đồng có; TODO |
| Code/report/hình | src + reports + outputs | Khung đầy đủ, kết quả phải sinh khi chạy |

## Giới hạn kiểm chứng

Đã kiểm cấu trúc, cú pháp, giao diện public, A/B/C/D, notebook JSON, lệnh check/help và ZIP sau giải nén. Đã kiểm check báo lỗi khi thiếu file hoặc lệch giao diện trên bản sao tạm. Chưa có PyTorch/Torchvision trong môi trường kiểm tra; chưa tải data, chưa train hoặc xác minh model reload/accuracy. requirements.txt chưa khóa phiên bản; TV6 chốt và chạy trên máy train rồi tạo lockfile. Chưa có GitHub repository được tạo.

Các thư mục outputs/runs và outputs/final sẽ được tạo khi chạy thật; không cần tạo CSV/PNG/PTH giả để lấp cấu trúc. Demo notebook thứ hai và confusion matrix là tùy chọn. Không cần thêm Docker, web app hay API để đáp ứng đề này.
