# FashionMNIST Lab 1 — Bộ khung nhóm 6 người

Đọc **docs/TEAM_CONTRACT.md trước khi code**, sau đó xem **docs/FILE_OWNERS.md** và tài liệu phân công trong docs.

## Trạng thái

Đã có: cấu trúc project, cấu hình A/B/C/D, kiểm tra cấu hình, CLI điều phối, seed/device, JSON/metadata, hàm nạp checkpoint và hợp đồng ghép code. Các phần dữ liệu, MLP, training, evaluate/visualize còn TODO để thành viên thực hiện. **Chưa tải data hoặc train model trong bộ khung này.** Không có metrics/checkpoint giả.

## Bắt đầu

1. Giải nén, mở thư mục fashionmnist-lab1 trong VS Code.
2. Kiểm tra bộ khung (chỉ cần Python):

```bash
python main.py --mode check
```

3. Tạo môi trường (Python có hỗ trợ bộ PyTorch/Torchvision được chọn):

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`.

4. Cài cặp PyTorch/Torchvision phù hợp CPU/CUDA theo https://pytorch.org/get-started/locally/ trên máy train. Sau đó:

```bash
python -m pip install -r requirements.txt
```

requirements.txt là danh sách thư viện, **chưa là môi trường khóa phiên bản**. TV6 chốt máy train rồi ghi phiên bản chạy thành công; sau khi môi trường sạch chạy được:

```bash
python -m pip freeze > requirements-lock.txt
```

Đưa requirements-lock.txt đã kiểm chứng vào Git. Không sao chép tùy tiện wheel/CUDA của máy khác.

## Lệnh tích hợp đã có giao diện

Các lệnh bên dưới chỉ hoàn thành khi các TODO tương ứng được triển khai. Hiện tại chúng báo rõ phần còn thiếu, không tạo kết quả giả.

```bash
# TV1: tải, kiểm tra, lưu chỉ số split chung và ảnh mẫu
python main.py --mode prepare --config configs/A.json

# TV1–4 hoàn thành rồi TV3 chạy baseline; TV5 thay A thành B/C/D
python main.py --mode train --config configs/A.json

# TV5: chọn bằng validation; main tự sao chép checkpoint do hàm chọn trả về
python main.py --mode select --config configs/A.json --runs A B C D

# TV4/TV6: đánh giá checkpoint đã chọn
# TV4 hoàn thành evaluate_final rồi chạy test cuối
python main.py --mode evaluate --checkpoint outputs/final/best_model.pth
```

Luôn chạy tại thư mục gốc. Tham số config/checkpoint tương đối được hiểu từ project root. Không ghi đè run đã tồn tại; nếu chạy lại, tạo config với run_id khác.

## Dành cho từng người

| Người | File chính | Việc khởi đầu |
|---|---|---|
| TV1 | src/data.py | Tích hợp 5 bước dữ liệu đã chạy; tạo prepare_data và build_dataloaders |
| TV2 | src/model.py | Hoàn thành MLP; không thay tên lớp/tham số |
| TV3 | src/train.py | Viết train_one_epoch/fit và checkpoint theo contract |
| TV4 | src/evaluate.py, src/visualize.py | Viết evaluate trước để TV3 ghép; sau đó hình/test cuối |
| TV5 | configs/*.json, src/experiments.py | Kiểm tra cấu hình; tổng hợp history và chọn run bằng validation |
| TV6 | main.py, src/utils.py, src/predict.py, docs | Tiếp nhận phần nền đã có, tích hợp, kiểm tra reload và báo cáo |

Mỗi người viết báo cáo ở reports/sections đúng phần của mình. Không đổi giao diện public mà chưa báo người sử dụng.

## Ghép sớm

Mốc đầu: một batch → forward → loss → backward/step → evaluate → save/reload. Đây là kiểm tra cần làm sau khi các TODO tương ứng hoàn thành, không phải kết quả đã đạt trong bộ khung.

Không dùng test để chọn config. TV5 chọn run trên validation trước, TV4 mới đánh giá test. Đọc docs/TEAM_CONTRACT.md về checkpoint/history/split.

## Kết quả double-check bộ khung

Xem docs/REVIEW_STATUS.md. Đã bổ sung select vào CLI, tự gọi loss plot sau fit, hàm check_reload và kiểm tra đầy đủ file/giao diện. Đây vẫn là scaffold giao nhiệm vụ; chuẩn bị dữ liệu, mô hình, train/evaluate/visualize và chọn run còn TODO. Cần cài và khóa môi trường trên máy thật trước khi chạy ML.

Mode check kiểm cấu hình A/B/C/D đúng kế hoạch ban đầu. Nếu nhóm chủ động đổi thiết kế thí nghiệm, TV6 phải cập nhật check và tài liệu cùng lúc.
