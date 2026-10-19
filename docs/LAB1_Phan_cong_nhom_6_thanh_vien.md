# LAB 1 — KẾ HOẠCH VÀ PHÂN CÔNG NHÓM 6 THÀNH VIÊN

**Đề tài:** PyTorch FashionMNIST Classification  
**Mục tiêu:** Xây dựng, huấn luyện, đánh giá và lưu–tải mạng nơ-ron phân loại ảnh thời trang bằng PyTorch.  
**Trạng thái tài liệu:** Kế hoạch thực hiện; chưa chứa kết quả thực nghiệm. Chỉ đánh dấu hoàn thành sau khi kiểm chứng thực tế.

## 1. Phạm vi và sản phẩm phải nộp

Theo Practice 1 trong tài liệu giảng viên, nhóm phải:

1. Tải FashionMNIST và áp dụng transforms.
2. Xây dựng mạng nơ-ron.
3. Viết vòng lặp huấn luyện: forward → loss → backward → cập nhật trọng số.
4. Đánh giá accuracy.
5. Lưu và tải lại mô hình.
6. Thử nghiệm kiến trúc hoặc hyperparameter.
7. Vẽ loss và hiển thị ảnh với nhãn dự đoán so với nhãn thật.
8. Nộp code Python, báo cáo ngắn, biểu đồ loss và ảnh kết quả.

**Các lựa chọn trong kế hoạch này** như MLP, cách chia 54.000/6.000, bốn cấu hình thử nghiệm, GitHub và tên file là đề xuất tổ chức của nhóm, không phải yêu cầu bắt buộc của giảng viên. Chưa có quy định trong đề về số trang, số slide hoặc ngưỡng accuracy; không tự đặt chúng thành điều kiện chấm điểm.

## 2. Điều kiện bắt đầu ghép các phần

Nhóm trưởng chạy và xác nhận 5 bước chuẩn bị dữ liệu: tải data → kiểm tra một ảnh → xem lưới ảnh → chia train/validation → tạo DataLoader.

- [ ] Tải được 60.000 ảnh train gốc và 10.000 ảnh test.
- [ ] Xem được ảnh và nhãn tương ứng.
- [ ] Một ảnh sau ToTensor có shape `[1, 28, 28]`, dtype `float32`, pixel trong `[0, 1]`.
- [ ] Batch đầy đủ với batch size 64 có shape ảnh `[64, 1, 28, 28]` và nhãn `[64]`.
- [ ] Nhãn số trong `[0, 9]`; batch nhãn có dtype `torch.int64`.
- [ ] Chia được 54.000 train / 6.000 validation, giữ nguyên 10.000 test.

Batch cuối có thể nhỏ hơn 64, đây không phải lỗi. Khi đạt các điều kiện trên, dữ liệu cơ bản sẵn sàng; thành viên 1 tiếp nhận và hoàn thiện phần kiểm tra, không cần tải/làm lại toàn bộ từ đầu. Các thành viên khác có thể chuẩn bị code song song trước khi ghép.

## 3. Dữ liệu và nguyên tắc xử lý

### 3.1 Nguồn dữ liệu

- Nguồn gốc: https://github.com/zalandoresearch/fashion-mnist
- API tải bằng PyTorch: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
- Tutorial giảng viên giới thiệu: https://docs.pytorch.org/tutorials/beginner/basics/intro.html

Dùng `torchvision.datasets.FashionMNIST(root="data", train=True/False, download=True, transform=...)`. Không phải tự gán nhãn, chuyển sang CSV hoặc thu thập một bộ ảnh khác.

| Nhãn | Tên lớp |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### 3.2 Có cần clean không?

Bản cơ bản sử dụng ảnh gốc. Cần kiểm tra số mẫu, kích thước, nhãn, pixel và phân bố lớp; không tự xóa ảnh mờ, ảnh khó hoặc sửa nhãn chỉ bằng cảm nhận. Không bắt buộc khử nhiễu, xóa nền, resize hoặc augmentation. `ToTensor()` là bước chuyển đổi đầu vào, không có nghĩa đã chuẩn hóa về mean 0/std 1.

Nếu thử thêm chuẩn hóa bằng thống kê dữ liệu, chỉ tính thống kê trên train rồi áp dụng cho validation/test. Nếu thử augmentation, chỉ áp dụng biến đổi ngẫu nhiên cho train và tách transforms của từng tập. Giữ bản cơ bản trước khi mở rộng.

### 3.3 Ngăn rò rỉ dữ liệu

- Train: cập nhật trọng số.
- Validation: chọn cấu hình và checkpoint.
- Test: chỉ đánh giá sau khi chốt cấu hình, không dùng để quyết định learning rate, số neuron hoặc epoch tốt nhất.
- Lưu chỉ số chia train/validation trong `outputs/split_indices.json`; kiểm tra hai tập chỉ số không giao nhau và phủ đủ 60.000 mẫu train gốc.
- Dùng chung chỉ số chia cho mọi run. Không chia lại theo seed khác giữa các cấu hình.
- Chia ngẫu nhiên không bảo đảm các lớp cân bằng tuyệt đối; cần đếm và ghi nhận thực tế.

## 4. Bảng phân công tổng quan

Điền tên thật trước khi bắt đầu.

| Thành viên | Họ tên | Trách nhiệm chính | Branch đề xuất | Phụ thuộc |
|---|---|---|---|---|
| TV1 | … | Dữ liệu, kiểm tra và DataLoader | `feature/data` | Nhận notebook dữ liệu ban đầu |
| TV2 | … | Mô hình MLP và giải thích kiến trúc | `feature/model` | Quy ước shape từ TV1 |
| TV3 | … | Training loop, history và checkpoint | `feature/training` | TV1, TV2, hàm evaluate của TV4 |
| TV4 | … | Evaluate, biểu đồ, ảnh dự đoán | `feature/evaluation` | TV1, TV2; history/checkpoint từ TV3/TV5 |
| TV5 | … | Cấu hình thử nghiệm và so sánh | `feature/experiments` | Bản cơ bản TV1–4 chạy đúng |
| TV6 | … | Tích hợp, lưu–tải, README, biên tập báo cáo | `feature/integration` | Nhận bàn giao tất cả các phần |

Mỗi người viết phần báo cáo tương ứng và hiểu được code mình phụ trách. TV6 tổng hợp, không chịu trách nhiệm viết thay nội dung của cả nhóm.

## 5. Quy ước kỹ thuật dùng chung

### 5.1 Cấu hình bản cơ bản

| Nội dung | Giá trị đề xuất |
|---|---|
| Mạng | Flatten → Linear(784,128) → ReLU → Linear(128,10) |
| Đầu vào | `[B, 1, 28, 28]` |
| Đầu ra | Logits `[B, 10]` |
| Nhãn | `[B]`, kiểu long, giá trị 0–9 |
| Loss | `torch.nn.CrossEntropyLoss()` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| Seed khởi tạo | 42 |
| Chọn checkpoint | Validation accuracy cao nhất; nếu bằng nhau, ưu tiên validation loss thấp hơn |
| Device | CUDA nếu có, nếu không dùng CPU |

Không thêm Softmax trước CrossEntropyLoss. Nhãn dự đoán lấy bằng `logits.argmax(dim=1)`. Softmax chỉ cần nếu muốn hiển thị xác suất.

### 5.2 Giao diện các module

Đây là hợp đồng nhóm cần thống nhất; thay đổi phải báo những người sử dụng hàm.

```python
# data.py
build_dataloaders(config)
# -> train_loader, val_loader, test_loader, class_names

# model.py
MLP(hidden_units=128)
# forward(images) -> logits [B, 10]

# train.py
train_one_epoch(model, loader, loss_fn, optimizer, device)
# -> {"loss": float, "accuracy": float}
fit(model, train_loader, val_loader, config, run_dir)
# -> dict: history_path, best_checkpoint_path, best_epoch

# evaluate.py
evaluate(model, loader, loss_fn, device)
# -> {"loss": float, "accuracy": float}
collect_predictions(model, loader, device)
# -> nhãn thật, nhãn dự đoán và chỉ số mẫu theo cùng thứ tự

# visualize.py
plot_history(history_csv, output_path)
plot_predictions(model, dataset, indices, class_names, device, output_path)

# predict.py
load_model(checkpoint_path, device)
# -> model đã nạp trọng số, metadata, ở chế độ eval
```

Lưu accuracy trong CSV/JSON ở thang **0–1**; chuyển sang phần trăm khi trình bày. Đường dẫn tính từ thư mục gốc project. Cùng một lần chạy, mọi biểu đồ, metrics và checkpoint phải có chung `run_id`.

## 6. Chi tiết nhiệm vụ từng thành viên

### TV1 — Dữ liệu và tiền xử lý

**Đầu vào:** notebook 5 bước đã chạy được, nguồn FashionMNIST, cấu hình chung.

**Việc thực hiện:**

1. Tiếp nhận và chuyển phần tải dữ liệu thành `src/data.py`.
2. Cài transforms cơ bản là ToTensor; giữ nguyên ảnh 28×28, chưa Flatten ở đây.
3. Tạo và lưu chỉ số chia train/validation; các run sau đọc lại chỉ số này.
4. Tạo DataLoader: train shuffle=True, validation/test shuffle=False.
5. Kiểm tra số mẫu, shape, dtype, pixel, nhãn và tính không giao nhau của chỉ số.
6. Đếm số mẫu từng lớp cho train/validation/test, lưu thành bảng.
7. Vẽ 15–20 ảnh có nhãn thật; chọn chỉ số cố định để chạy lại được.
8. Viết giải thích về dữ liệu, transforms, split và lý do chưa cần clean sâu.

**Bàn giao:** `src/data.py`, `outputs/split_indices.json`, `outputs/data_summary.json`, `outputs/class_distribution.csv`, `outputs/sample_images.png`, `reports/sections/01_data.md`.

**Tiêu chí hoàn thành:** gọi được build_dataloaders; tổng kích thước đúng; chỉ số split hợp lệ; có ảnh minh họa và bảng kiểm tra. Bàn giao cách sử dụng cho TV2/TV3/TV4.

**Câu hỏi cần giải thích:** Dataset khác DataLoader thế nào? `[B,1,28,28]` là gì? Vì sao train shuffle còn validation/test không cần? ToTensor làm gì?

### TV2 — Kiến trúc mạng MLP

**Đầu vào:** hợp đồng tensor, số lớp và cấu hình mạng.

**Việc thực hiện:**

1. Viết lớp MLP kế thừa nn.Module; khai báo lớp trong __init__, tính toán trong forward.
2. Cho phép thay hidden_units bằng cấu hình, tối thiểu hỗ trợ 128 và 256.
3. Dùng Flatten bên trong mô hình; đưa ra 10 logits.
4. Chạy kiểm tra forward bằng một batch hoặc tensor thử, xác nhận `[B,10]` và giá trị hữu hạn.
5. In cấu trúc và tính số tham số có thể học.
6. Lập bảng shape qua từng lớp; giải thích Linear, ReLU, logits và argmax.
7. Viết phần model cho báo cáo và chuyển ví dụ khởi tạo cho TV3/TV6.

**Bàn giao:** `src/model.py`, `reports/sections/02_model.md`, bảng kiến trúc và số tham số.

**Tiêu chí hoàn thành:** model nhận `[B,1,28,28]`, trả `[B,10]`; hỗ trợ hai quy mô hidden; không hard-code batch size; không thêm Softmax trước loss.

**Câu hỏi cần giải thích:** Vì sao 784 đầu vào? Vì sao 10 đầu ra? ReLU có vai trò gì? Parameter khác hyperparameter thế nào?

### TV3 — Huấn luyện và checkpoint

**Đầu vào:** DataLoader TV1, model TV2, evaluate TV4, config chung.

**Việc thực hiện:**

1. Viết train_one_epoch và fit; chuyển cả ảnh, nhãn và mô hình về cùng device.
2. Với mỗi epoch, gọi model.train(); mỗi batch thực hiện zero_grad → forward → loss → backward → optimizer.step.
3. Cộng loss theo số mẫu: `sum(batch_loss * batch_size) / total_samples`; accuracy là tổng dự đoán đúng / tổng mẫu.
4. Sau mỗi epoch, gọi evaluate trên validation và ghi log.
5. Ghi history CSV gồm epoch, train_loss, train_accuracy, val_loss, val_accuracy; có thể thêm thời gian.
6. Lưu checkpoint tốt nhất theo quy tắc đã thống nhất; không mặc định checkpoint cuối là tốt nhất.
7. Thống nhất định dạng checkpoint với TV6, chứa trọng số và metadata cấu hình.
8. Chạy kiểm tra ngắn với ít batch trước, sau đó chạy baseline đủ cấu hình; kiểm tra trọng số thực sự thay đổi và loss không NaN/Inf.
9. Viết phần training, autograd, optimizer cho báo cáo.

**Bàn giao:** `src/train.py`, history và best checkpoint của baseline trong `outputs/runs/A/`, `reports/sections/03_training.md`.

**Tiêu chí hoàn thành:** bản cơ bản chạy đủ epoch; log đầy đủ, loss/accuracy tính đúng theo mẫu; khôi phục model.train sau các lượt validation; checkpoint đúng epoch được chọn.

**Câu hỏi cần giải thích:** Vì sao xóa gradient? backward khác step thế nào? Epoch khác batch thế nào? Loss khác accuracy thế nào?

### TV4 — Đánh giá và hình kết quả

**Đầu vào:** DataLoader/model; history từ TV3/TV5; checkpoint cuối được TV5 chọn.

**Việc thực hiện:**

1. Viết evaluate sớm để TV3 sử dụng: model.eval và torch.no_grad; tuyệt đối không optimizer.step trong hàm này.
2. Tính loss/accuracy trên toàn bộ tập được truyền vào, theo số mẫu thực tế.
3. Viết collect_predictions, giữ đúng thứ tự ảnh và nhãn; không ghép nhãn của batch này với ảnh batch khác.
4. Vẽ train/validation loss theo epoch cho từng run; ghi trục, tiêu đề, legend và run_id.
5. Có thể bổ sung biểu đồ accuracy; bắt buộc có loss theo đề.
6. Sau khi TV5 chốt cấu hình, nạp best checkpoint rồi đánh giá trên test; lưu số mẫu, loss, accuracy, run_id và checkpoint nguồn.
7. Vẽ lưới ảnh test có Actual/Predicted, đánh dấu đúng/sai; thêm lưới lỗi nếu có, không giả tạo mẫu sai.
8. Phân tích một số lỗi quan sát được; không kết luận lớp nào hay nhầm nếu chưa có bằng chứng.
9. Confusion matrix là phần mở rộng tùy thời gian.

**Bàn giao:** `src/evaluate.py`, `src/visualize.py`, loss_curves.png từng run, `outputs/final/test_metrics.json`, `predictions.png`, `misclassified.png` nếu có lỗi, `reports/sections/04_evaluation.md`.

**Tiêu chí hoàn thành:** không cập nhật trọng số khi đánh giá; ảnh–nhãn đúng cặp; biểu đồ đọc được; metrics cuối tương ứng đúng checkpoint đã chọn.

**Câu hỏi cần giải thích:** Vì sao cần eval và no_grad? Vì sao test không dùng để chọn cấu hình? Loss giảm nhưng accuracy có luôn tăng không?

### TV5 — Thực nghiệm và lựa chọn mô hình

**Đầu vào:** baseline chạy đúng, bộ split cố định và các module TV1–4.

**Việc thực hiện:**

1. Chuẩn bị cấu hình trước khi chạy theo bảng dưới; không sửa code lõi tùy tiện cho từng run.
2. Mỗi run khởi tạo model và optimizer mới; đặt lại seed Python/NumPy/PyTorch, dùng cùng split và cùng quy tắc shuffle.
3. Giữ cùng số epoch, transforms, optimizer và quy tắc chọn checkpoint khi so sánh learning rate/hidden units.
4. Chạy các cấu hình; tái sử dụng kết quả A từ TV3 nếu đáp ứng đúng quy ước và đủ metadata, không phải chạy lại vô ích.
5. Lưu config, history, checkpoint và môi trường của mỗi run. Seed giúp tái lập nhưng không bảo đảm kết quả tuyệt đối giống nhau giữa mọi phần cứng.
6. Ghi lỗi hoặc run không hội tụ như kết quả thực tế; không sửa/xóa số để làm bảng đẹp.
7. Chọn run tốt nhất bằng validation theo quy tắc chung; bàn giao chính xác đường dẫn best checkpoint cho TV4/TV6.
8. Viết so sánh và giải thích dựa trên dữ liệu; một seed chỉ cho so sánh ban đầu, chưa chứng minh ưu thế thống kê.

| Run | Hidden units | Learning rate | Batch size | Epochs | Câu hỏi |
|---|---:|---:|---:|---:|---|
| A | 128 | 0.001 | 64 | 10 | Baseline học thế nào? |
| B | 128 | 0.0001 | 64 | 10 | Learning rate nhỏ hơn tác động gì? |
| C | 128 | 0.01 | 64 | 10 | Learning rate lớn hơn tác động gì? |
| D | 256 | 0.001 | 64 | 10 | Tăng số neuron có cải thiện không? |

**Bàn giao:** `configs/A.json` đến `D.json`, `outputs/experiment_results.csv`, thư mục kết quả từng run, `reports/sections/05_experiments.md`.

CSV tổng hợp tối thiểu: run_id, hidden_units, learning_rate, batch_size, epochs, seed, best_epoch, best_val_accuracy, best_val_loss, duration_seconds, device, commit_sha. Nếu so thời gian, phải ghi rõ khác biệt phần cứng; không quy toàn bộ chênh lệch thời gian cho mô hình.

**Tiêu chí hoàn thành:** mỗi dòng kết quả truy ngược được về cấu hình/history/checkpoint; chọn bằng validation; không bịa ngưỡng accuracy hoặc kết quả kỳ vọng.

**Câu hỏi cần giải thích:** Vì sao thay từng yếu tố? Vì sao khởi tạo lại model? Dấu hiệu overfitting trên train/validation là gì?

### TV6 — Tích hợp, lưu–tải và đóng gói

**Đầu vào:** các module, cấu hình, kết quả và phần báo cáo của TV1–5.

**Việc thực hiện:**

1. Tạo repo, thư mục, README và .gitignore từ đầu; thống nhất hợp đồng hàm với cả nhóm.
2. Ghi dependencies và phiên bản đã dùng; không tự chọn phiên bản CUDA nếu chưa biết máy train.
3. Tạo `main.py` để gọi các phần, cùng notebook minh họa nếu cần. Các lệnh trong README phải là lệnh đã chạy thật.
4. Thống nhất với TV3 định dạng checkpoint: model_state_dict, model_config, preprocessing, class_names, run_id, epoch, val_metrics.
5. Viết `src/predict.py`: đọc metadata, tạo đúng kiến trúc, tải trọng số với map_location phù hợp và gọi eval.
6. Kiểm tra trên cùng một đầu vào: logits trước/sau lưu gần bằng nhau với dung sai hợp lý; nhãn dự đoán giống nhau. Ghi kết quả kiểm tra, không chỉ chứng minh file tồn tại.
7. Tập hợp checkpoint được chọn và metrics/hình từ TV4; không dùng nhầm checkpoint cuối epoch.
8. Tổng hợp báo cáo từ các phần của từng thành viên; thống nhất tên lớp, đơn vị accuracy, cấu hình và số liệu.
9. Nhờ một người khác chạy theo README trong thư mục/môi trường sạch, tải dữ liệu từ đầu nếu cần; sửa hướng dẫn nếu không tái lập được.
10. Kiểm tra gói nộp cuối có code, report, loss graphs, image displays và checkpoint theo cách nộp đã thống nhất.

**Bàn giao:** `main.py`, `src/predict.py`, `README.md`, `requirements.txt`, `.gitignore`, `reports/Lab1_Report.md`, kết quả kiểm tra lưu–tải và hướng dẫn lấy checkpoint.

**Tiêu chí hoàn thành:** người khác làm theo README chạy được; model tải lại dự đoán đúng quy trình; tài liệu và kết quả thống nhất; không ghi đè dữ liệu hoặc file kết quả ngoài ý muốn.

**Câu hỏi cần giải thích:** Lưu state_dict là lưu gì? Vì sao phải lưu kiến trúc/transforms? Làm sao chứng minh mô hình tải lại hoạt động?

## 7. Cấu trúc project cụ thể và người phụ trách từng file

### 7.1 Cách đọc cấu trúc

- Thư mục gốc project đặt tên `fashionmnist-lab1/`.
- **[VIẾT]**: thành viên phải tạo hoặc chỉnh nội dung file.
- **[SINH TỰ ĐỘNG]**: chương trình tạo khi chạy, không gõ số liệu hoặc tạo checkpoint bằng tay.
- **[TẢI]**: thư viện tải dữ liệu về, không chỉnh sửa thủ công.
- **[TÙY CHỌN]**: bổ sung khi có thời gian, không phải yêu cầu bắt buộc của đề.
- Chủ file chịu trách nhiệm chính; người phối hợp review và cung cấp đầu vào. Khi cần sửa file của người khác, trao đổi trước.

### 7.2 Cây thư mục có chú thích

Danh sách lồng nhau bên dưới thể hiện cấu trúc thư mục. `A/` là ví dụ một run; B/C/D có cấu trúc tương tự. Đây là cấu trúc **cần xây dựng**, không khẳng định các file code hoặc kết quả đã tồn tại.

- **`fashionmnist-lab1/`** — thư mục gốc; TV6 khởi tạo.
  - `README.md` — **[VIẾT · TV6]** cách cài, chạy, xem kết quả và lấy checkpoint.
  - `requirements.txt` — **[VIẾT · TV6]** dependencies và phiên bản môi trường đã kiểm tra.
  - `.gitignore` — **[VIẾT · TV6]** bỏ data, cache, môi trường ảo và checkpoint khỏi Git theo thỏa thuận.
  - `main.py` — **[VIẾT · TV6]** đọc cấu hình, điều phối các module và chế độ chạy.
  - **`configs/`** — cấu hình thí nghiệm; TV5 quản lý.
    - `A.json` — **[VIẾT · TV5, TV3 review]** baseline: hidden=128, lr=0.001.
    - `B.json` — **[VIẾT · TV5]** hidden=128, lr=0.0001.
    - `C.json` — **[VIẾT · TV5]** hidden=128, lr=0.01.
    - `D.json` — **[VIẾT · TV5]** hidden=256, lr=0.001.
  - **`src/`** — code chính; mỗi file có một chủ sở hữu.
    - `__init__.py` — **[VIẾT · TV6]** có thể để trống, đánh dấu package.
    - `data.py` — **[VIẾT · TV1]** tải, transforms, split, DataLoader và kiểm tra dữ liệu.
    - `model.py` — **[VIẾT · TV2]** định nghĩa MLP và forward.
    - `train.py` — **[VIẾT · TV3]** train_one_epoch, fit, history và lưu best checkpoint.
    - `evaluate.py` — **[VIẾT · TV4]** loss, accuracy và thu thập dự đoán.
    - `visualize.py` — **[VIẾT · TV4]** biểu đồ loss và lưới ảnh Actual/Predicted.
    - `predict.py` — **[VIẾT · TV6]** tạo lại model từ metadata, nạp trọng số và inference.
    - `utils.py` — **[VIẾT · TV6]** seed dùng chung, đọc/ghi JSON và thu thập metadata môi trường.
    - `experiments.py` — **[VIẾT · TV5]** tổng hợp các run và chọn checkpoint bằng validation.
  - **`notebooks/`** — phần minh họa; giữ nhẹ, gọi code từ src.
    - `01_data_check.ipynb` — **[VIẾT · TV1]** tiếp nhận 5 bước đã chạy và kiểm tra dữ liệu.
    - `02_lab1_demo.ipynb` — **[TÙY CHỌN · VIẾT · TV6]** demo toàn bộ quy trình bằng các module đã ghép.
  - **`data/`** — dữ liệu tải về, TV1 quản lý; không commit.
    - **`FashionMNIST/raw/`** — **[TẢI]** ảnh/nhãn gốc và file giải nén do Torchvision quản lý; không tự đổi tên. Chi tiết file phụ thuộc phiên bản thư viện.
  - **`outputs/`** — kết quả thực tế, không điền số minh họa vào đây.
    - `split_indices.json` — **[SINH TỰ ĐỘNG · TV1]** chỉ số train/validation để dùng lại.
    - `data_summary.json` — **[SINH TỰ ĐỘNG · TV1]** số mẫu, shape, dtype và kiểm tra dữ liệu.
    - `class_distribution.csv` — **[SINH TỰ ĐỘNG · TV1]** số ảnh mỗi lớp của từng tập.
    - `sample_images.png` — **[SINH TỰ ĐỘNG · TV1]** lưới ảnh có nhãn thật.
    - `experiment_results.csv` — **[SINH TỰ ĐỘNG · TV5]** tổng hợp các run từ history/metadata.
    - **`runs/A/`** — TV3 tạo baseline; TV5 quản lý các run B/C/D tương tự.
      - `config.json` — **[SINH TỰ ĐỘNG · TV3]** cấu hình hiệu lực của lần chạy.
      - `metadata.json` — **[SINH TỰ ĐỘNG · TV3, TV6 hỗ trợ]** commit, seed, device, phiên bản và thời gian chạy.
      - `history.csv` — **[SINH TỰ ĐỘNG · TV3]** loss/accuracy mỗi epoch.
      - `best_model.pth` — **[SINH TỰ ĐỘNG · TV3]** checkpoint tốt nhất theo validation.
      - `loss_curves.png` — **[SINH TỰ ĐỘNG · TV4]** vẽ từ history của đúng run.
      - `accuracy_curves.png` — **[TÙY CHỌN · SINH TỰ ĐỘNG · TV4]** train/validation accuracy.
    - **`final/`** — chỉ tạo sau khi TV5 chọn run cuối.
      - `selection.json` — **[SINH TỰ ĐỘNG · TV5]** run được chọn, tiêu chí, best epoch và checkpoint nguồn.
      - `best_model.pth` — **[SINH TỰ ĐỘNG · TV6]** bản sao checkpoint được chọn, không phải train thêm.
      - `test_metrics.json` — **[SINH TỰ ĐỘNG · TV4]** test loss/accuracy, số mẫu và checkpoint nguồn.
      - `test_predictions.csv` — **[SINH TỰ ĐỘNG · TV4]** test_index, true_label, predicted_label của mọi ảnh test.
      - `predictions.png` — **[SINH TỰ ĐỘNG · TV4]** ảnh với Actual/Predicted.
      - `misclassified.png` — **[SINH TỰ ĐỘNG · TV4]** ví dụ sai, chỉ tạo nếu có mẫu sai.
      - `reload_check.json` — **[SINH TỰ ĐỘNG · TV6]** kết quả so sánh trước/sau lưu–tải.
  - **`reports/`** — nội dung báo cáo.
    - **`sections/`** — mỗi người viết phần của mình.
      - `01_data.md` — **[VIẾT · TV1]** dữ liệu và tiền xử lý.
      - `02_model.md` — **[VIẾT · TV2]** mô hình và số tham số.
      - `03_training.md` — **[VIẾT · TV3]** quy trình học và checkpoint.
      - `04_evaluation.md` — **[VIẾT · TV4]** metrics, biểu đồ và lỗi dự đoán.
      - `05_experiments.md` — **[VIẾT · TV5]** so sánh cấu hình và chọn run.
      - `06_integration.md` — **[VIẾT · TV6]** tái lập môi trường, lưu–tải và cách sử dụng.
    - `Lab1_Report.md` — **[VIẾT · TV6 biên tập, cả nhóm review]** báo cáo cuối, liên kết đúng ảnh từ outputs.

TV1 có thể gọi các hàm vẽ chung từ TV4 để tạo sample_images.png; TV1 vẫn chịu trách nhiệm nội dung và đầu ra của phần dữ liệu. Không bắt buộc tạo demo notebook nếu main.py và báo cáo đã trình bày đầy đủ.

### 7.3 Chi tiết code cần viết trong mỗi file nguồn

| File | Chủ file | Nội dung phải triển khai | Đầu vào → đầu ra | Ai sử dụng? |
|---|---|---|---|---|
| `src/data.py` | TV1 | Tải dataset, transforms, lưu/đọc split_indices, kiểm tra split, build_dataloaders; hỗ trợ hàm tạo summary/phân bố | config → 3 loader + class_names | TV3, TV4, TV6 |
| `src/model.py` | TV2 | MLP(hidden_units), __init__, forward; hàm hoặc đoạn kiểm tra số tham số | ảnh `[B,1,28,28]` → logits `[B,10]` | TV3, TV4, TV6 |
| `src/train.py` | TV3 | train_one_epoch, fit; gọi evaluate từ TV4; ghi history; chọn và lưu checkpoint theo hợp đồng | model + loader + config + run_dir → history + best checkpoint | TV5, TV6 |
| `src/evaluate.py` | TV4 | evaluate và collect_predictions; eval/no_grad; tính theo tổng số mẫu | model + loader → metrics, nhãn thật/dự đoán/chỉ số | TV3, TV5, TV6 |
| `src/visualize.py` | TV4 | plot_history, plot_predictions; đóng figure sau khi lưu; không thay đổi trọng số | history hoặc model/dataset → PNG | TV1, TV5, TV6 |
| `src/predict.py` | TV6 | load_model và hàm dự đoán; map_location, class_names, eval | checkpoint + ảnh → logits/nhãn | main, notebook demo |
| `src/utils.py` | TV6 | set_seed, load_json, save_json, collect_environment; ghi metadata không chứa token | seed/path/object → môi trường đã đặt seed hoặc JSON | TV1, TV3, TV5, main |
| `main.py` | TV6 | Đọc CLI/config, kiểm tra đường dẫn, tạo model/loss/optimizer và gọi module đúng thứ tự; chế độ prepare/train/evaluate | tham số chạy → tác vụ và kết quả | Cả nhóm |

**Quy tắc phụ thuộc:** data.py/model.py/utils.py không import main.py. train.py được import evaluate.py; evaluate.py không import ngược train.py. Các module không tự tải data hoặc train ngay khi được import; việc chạy do main.py/notebook điều phối. Tránh circular import và tránh chạy train ngoài ý muốn.

### 7.4 Cấu hình JSON và cách điều phối

TV5 viết A.json theo mẫu sau; B/C/D chỉ thay run_id và thông số cần khảo sát:

```json
{
  "run_id": "A",
  "data_root": "data",
  "split_indices_path": "outputs/split_indices.json",
  "train_size": 54000,
  "val_size": 6000,
  "seed": 42,
  "batch_size": 64,
  "num_workers": 0,
  "transform": "to_tensor",
  "model_name": "mlp",
  "hidden_units": 128,
  "optimizer": "adam",
  "learning_rate": 0.001,
  "epochs": 10,
  "device": "auto",
  "output_root": "outputs"
}
```

`num_workers=0` là cấu hình ban đầu đơn giản cho notebook/Windows; chỉ thay sau khi kiểm tra môi trường. `device=auto` cần TV6 triển khai chọn CUDA nếu khả dụng, nếu không chọn CPU. Không ghi chuỗi auto trực tiếp vào torch.device. Bộ split dùng chung nên chuẩn bị một lần trước khi các máy chạy song song.

**Giao diện lệnh TV6 cần triển khai** (đây là mục tiêu thiết kế, chưa phải lệnh đã chạy thành công):

```bash
# TV1/TV6: tải, kiểm tra và chuẩn bị split chung.
python main.py --mode prepare --config configs/A.json

# TV3: baseline. TV5 chạy B/C/D với config tương ứng.
python main.py --mode train --config configs/A.json

# TV4/TV6: chỉ chạy sau khi chọn run và tạo checkpoint cuối.
python main.py --mode evaluate --checkpoint outputs/final/best_model.pth
```

TV6 phải hiện thực và kiểm tra các lệnh trước khi đưa vào README như hướng dẫn chính thức. Evaluate đọc preprocessing/model_config từ checkpoint, dùng đường dẫn data mặc định của project hoặc tham số riêng đã ghi trong README. Chỉ dùng test ở chế độ đánh giá cuối.

### 7.5 Dữ liệu bàn giao giữa các file

| Bên giao → bên nhận | File/thông tin | Cách kiểm tra bàn giao |
|---|---|---|
| TV1 → TV3/TV4 | data.py, split_indices.json, class_names | Batch đúng shape; dùng lại đúng split |
| TV2 → TV3/TV6 | model.py, hidden_units và số lớp | Forward ra `[B,10]`; tạo lại đúng kiến trúc |
| TV4 → TV3 | evaluate.py | Trả loss/accuracy 0–1; không update trọng số |
| TV3 → TV5/TV4 | run A: config, history, checkpoint, metadata | history đủ epoch; checkpoint khớp epoch được chọn |
| TV5 → TV4/TV6 | experiment_results.csv và final/selection.json | Chỉ rõ run_id, best_epoch, tiêu chí và checkpoint_path |
| TV4 → TV6 | test_metrics, test_predictions, PNG | Mọi output khớp checkpoint nguồn và thứ tự ảnh |
| TV6 → cả nhóm | README, main.py, reload_check, report | Một người khác chạy lại được và xác nhận |

**Nội dung checkpoint tối thiểu TV3 và TV6 phải thống nhất:** model_state_dict, model_config, preprocessing, class_names, run_id, epoch, val_metrics. Optimizer state chỉ cần thêm nếu nhóm muốn tiếp tục huấn luyện. Checkpoint best được lưu sau validation của đúng epoch; tên file giống nhau không đủ để chứng minh cùng mô hình, cần metadata nguồn.

**Cột history.csv:** epoch, train_loss, train_accuracy, val_loss, val_accuracy. **Cột test_predictions.csv:** test_index, true_label, predicted_label; test loader không shuffle nên có thể giữ thứ tự chỉ số gốc.

### 7.6 Quy tắc tạo và quản lý file

1. TV6 tạo bộ khung thư mục và các file nguồn trống; sau đó mỗi người triển khai đúng file sở hữu. Không tạo file kết quả giả để lấp cây thư mục.
2. TV5 tạo A.json sớm để TV1/TV3 có config chạy baseline; không cần đợi baseline xong mới giao cấu hình.
3. TV3 viết cơ chế sinh history/checkpoint dùng chung. TV5 chạy cơ chế đó cho B/C/D, không phải tự viết lại vòng lặp train.
4. Mỗi run có thư mục riêng. Nếu cần chạy lại, dùng run_id mới như A_repeat1 hoặc lưu bản cũ trước; không âm thầm ghi đè kết quả đã đưa vào báo cáo.
5. Checkpoint được bỏ khỏi Git theo .gitignore nhưng vẫn phải lưu và bàn giao ở kênh nhóm thống nhất. README ghi cách lấy và đặt vào đúng đường dẫn.
6. Commit code, configs, reports và các CSV/JSON/PNG nhỏ đã chọn. Không đưa toàn bộ outputs vào .gitignore nếu nhóm muốn theo dõi các kết quả nhỏ này trên Git.
7. Nếu không có misclassified.png vì không có lỗi, ghi rõ trong báo cáo; không tạo ảnh sai giả. Các file tùy chọn không là điều kiện hoàn thành bắt buộc.
8. Báo cáo cuối phải tham chiếu ảnh bằng đường dẫn tương đối hợp lệ; ví dụ từ reports/Lab1_Report.md đến ảnh là `../outputs/final/predictions.png`.

### 7.7 Checklist nhanh theo người

| Người | File phải viết chính | File kết quả chịu trách nhiệm |
|---|---|---|
| TV1 | data.py, 01_data_check.ipynb, 01_data.md | split_indices, data_summary, class_distribution, sample_images |
| TV2 | model.py, 02_model.md | Kiểm tra shape, bảng kiến trúc/số tham số trong phần báo cáo |
| TV3 | train.py, 03_training.md | Cơ chế tạo config/history/metadata/checkpoint và kết quả A |
| TV4 | evaluate.py, visualize.py, 04_evaluation.md | loss graphs mọi run; test_metrics, test_predictions, ảnh cuối |
| TV5 | A/B/C/D.json, 05_experiments.md | B/C/D qua train chung; experiment_results, selection.json |
| TV6 | main.py, utils.py, predict.py, __init__.py, README, requirements, .gitignore, 06_integration.md, Lab1_Report.md | Checkpoint cuối, reload_check; demo notebook nếu cần |

## 8. Trình tự phối hợp và các mốc

| Mốc | Công việc | Điều kiện xong |
|---|---|---|
| M0 — Data sẵn sàng | Nhóm trưởng chạy 5 bước; TV1 xác nhận | Có batch/ảnh/nhãn đúng |
| M1 — Hợp đồng chung | TV6 tạo repo; cả nhóm chốt API, cấu hình, split | Các hàm và file có người sở hữu |
| M2 — Ghép bản ngắn | TV1+TV2+TV3+TV4 ghép; chạy ít batch | Forward/backward/evaluate không lỗi |
| M3 — Baseline đầy đủ | TV3 chạy A, TV4 vẽ, TV6 thử lưu–tải | Có history, checkpoint và loss graph |
| M4 — Thực nghiệm | TV5 chạy B/C/D, TV4 vẽ so sánh | Bảng validation đủ và chọn run cuối |
| M5 — Đánh giá cuối | TV4 test checkpoint đã chọn; TV6 xác nhận reload | Có test metrics và Actual/Predicted |
| M6 — Đóng gói | Mỗi người viết phần mình; TV6 tổng hợp, người khác chạy README | Gói nộp đầy đủ, chạy lại được |

Trong M1–M2: TV1/TV2 làm song song; TV3/TV4 viết theo hợp đồng hàm; TV5 chuẩn bị cấu hình và bảng kết quả; TV6 chuẩn bị repo và báo cáo. Không cần chờ từng người hoàn thành hết mới bắt đầu việc tiếp theo.

## 9. GitHub và máy train dùng chung

1. Một repository, sáu tài khoản thành viên; tạo một issue cho mỗi phần.
2. Mỗi người làm trên branch riêng, commit rõ nội dung và mở pull request.
3. Ít nhất một người khác review giao diện hàm và kết quả kiểm tra trước khi merge.
4. Hạn chế cùng sửa một notebook; code chính chia module, notebook gọi lại module.
5. Chốt một commit trước khi train; lưu commit SHA trong metadata. Không thay code đang được một tiến trình train sử dụng.
6. Dùng VS Code để code; Live Share khi cần debug chung; máy một thành viên hoặc Kaggle có thể là nơi chạy.
7. Cùng truy cập terminal không cộng sức mạnh GPU của sáu máy. Với tài nguyên hạn chế, xếp lịch chạy A/B/C/D và tránh ghi cùng thư mục.
8. Không commit `data/`, môi trường ảo, cache hoặc token. Checkpoint dùng kênh lưu đã thống nhất và ghi đường dẫn trong README; các file nhỏ như CSV/JSON/PNG chọn lọc có thể commit.
9. Nếu chạy cloud, tải/lưu kết quả ra nơi bền vững trước khi phiên kết thúc.

**Mẫu PR:** Mục tiêu → file thay đổi → cách chạy kiểm tra → bằng chứng output → người nhận bàn giao.

## 10. Nội dung báo cáo ngắn

| Phần | Nội dung | Người viết gốc |
|---|---|---|
| Giới thiệu | Bài toán, đầu vào/đầu ra, mục tiêu | TV6 tổng hợp |
| Dữ liệu | Nguồn, lớp, transforms, split, kiểm tra | TV1 |
| Mô hình | Kiến trúc, shape, số tham số | TV2 |
| Huấn luyện | Loss, optimizer, forward/backward/update | TV3 |
| Thực nghiệm | Bảng cấu hình, kết quả validation, cách chọn | TV5 |
| Đánh giá | Test accuracy/loss, biểu đồ, ảnh đúng/sai | TV4 |
| Lưu–tải và tái lập | Checkpoint, môi trường, cách chạy | TV6 |
| Kết luận | Kết quả quan sát, hạn chế, hướng cải thiện | Cả nhóm |

Không biến giả thuyết thành kết luận. Ví dụ “mạng lớn hơn có thể tốt hơn” phải kiểm tra bằng run A và D; chỉ ghi “tốt hơn trong lần chạy này” nếu số liệu thực tế hỗ trợ.

## 11. Checklist nghiệm thu cuối

- [ ] Data: nguồn đúng, shape/nhãn đúng, có ảnh mẫu và kiểm tra split.
- [ ] Transforms: được giải thích, không dùng thống kê test để xử lý train.
- [ ] Model: nhận ảnh và trả 10 logits; giải thích được kiến trúc.
- [ ] Training: đủ forward, loss, backward, optimize; ghi history.
- [ ] Evaluation: không cập nhật trọng số; accuracy tính theo số mẫu.
- [ ] Experiments: có cấu hình và số liệu thật; chọn bằng validation.
- [ ] Loss graphs: có nhãn trục/legend, đúng run và history nguồn.
- [ ] Image displays: có ảnh, nhãn thật và nhãn dự đoán.
- [ ] Save/load: checkpoint nạp vào mô hình mới và kiểm tra dự đoán.
- [ ] Test cuối: khớp run/checkpoint được chọn, chưa dùng để tuning.
- [ ] Code: chạy lại được theo README; dependencies được ghi rõ.
- [ ] Report: số liệu thống nhất với CSV, biểu đồ và checkpoint.
- [ ] Bàn giao: người nhận xác nhận dùng được phần của người trước.
- [ ] Gói nộp: code Python, báo cáo ngắn, loss graphs, image displays; checkpoint/hướng dẫn lấy được gửi kèm theo thỏa thuận.

## 12. Mẫu cập nhật tiến độ

Mỗi thành viên dùng mẫu sau ở issue của mình:

```text
Thành viên:
Nhiệm vụ:
Đã hoàn thành:
File / branch / pull request:
Cách kiểm tra và output quan sát:
Phần còn lại:
Vướng mắc / cần ai hỗ trợ:
Đã bàn giao cho ai:
```

**Nguyên tắc hoàn thành:** có code + output thật + giải thích + người tiếp nhận sử dụng được. Không coi việc gửi một file chưa chạy hoặc một ảnh chụp màn hình riêng lẻ là đã bàn giao đầy đủ.


## 13. Bộ khung triển khai đã chốt

Đã chuẩn bị gói fashionmnist-lab1 với CLI, config và các module có TODO. Đọc `docs/TEAM_CONTRACT.md` trong gói trước khi triển khai; đây là nguồn ưu tiên về giao diện ghép code. `fit()` trả dict gồm history_path, best_checkpoint_path, best_epoch; `load_model()` trả model và metadata. Main chọn device/seed, tạo model/run_dir; fit tạo loss/optimizer. TV5 có thêm `src/experiments.py` để tổng hợp và chọn run; TV6 quản lý `src/utils.py`. Hàm `evaluate_final()` thuộc TV4. Bộ khung chưa tải dữ liệu hoặc train; các phần này chờ thực hiện và xác minh trên máy nhóm.


## 14. Double-check cấu trúc và luồng tích hợp

Đã rà soát lại: project đủ các phần cho đề Lab 1. Bộ khung có CLI check/prepare/train/select/evaluate; mode select gọi TV5 và sao chép checkpoint cuối, train nối hàm vẽ loss sau fit. predict.py có check_reload để kiểm tra thực tế sau khi thuật toán được triển khai. Mode check kiểm danh sách file, config, cú pháp và chữ ký hàm nhưng không thay việc chạy ML. Xem docs/REVIEW_STATUS.md trong ZIP về các điểm sửa và giới hạn xác minh. Còn phải cài/khóa PyTorch trên máy train, thực hiện các TODO và ghép một batch. Không có kết quả train/data/accuracy giả trong project.
