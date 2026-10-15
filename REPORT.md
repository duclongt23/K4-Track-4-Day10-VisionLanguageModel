Họ tên: Nguyễn Đức Long
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0,095 | 0,071–0,123 | 1,11 | 0,132 | 0,033 | 0,200 |
| E1 | LoRA LLM | 33.030.144 | 10,475 | 45,16 | 0,252 | 0,215–0,289 | 1,21 | 0,144 | 0,182 | 0,524 |
| E2 | + projector | 142.135.296 | 11,469 | 45,61 | 0,325 | 0,286–0,367 | 1,04 | 0,189 | 0,274 | 0,558 |
| E3 | + LoRA ViT | 148.426.752 | 11,650 | 51,59 | 0,432 | 0,390–0,475 | 0,79 | 0,302 | 0,371 | 0,646 |
| E4 | data_size = 0,1 | 148.426.752 | 11,096 | 50,59 | 0,263 | 0,230–0,299 | 1,73 | 0,177 | 0,181 | 0,570 |
| E4 | data_size = 1,0 (chính là E3) | 148.426.752 | 11,650 | 51,59 | 0,432 | 0,390–0,475 | 0,79 | 0,302 | 0,371 | 0,646 |
| E5 | GRPO từ E3 (60 bước) | 148.426.752 | 6,317 | 32,63 | 0,439 | 0,395–0,480 | 0,63 | 0,307 | 0,365 | 0,674 |

Tất cả chấm trên 311 ảnh test, thang 0–1000. `parse_valid_rate`: E0 0,881; E1, E2, E3 đều 1,000; E4 (10%) 0,997; E5 1,000.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Model có khoảng 4,44 tỉ tham số: LLM (cả embedding) khoảng 4,0 tỉ, ViT khoảng 415 triệu. Projector nằm trong ViT: `visual.merger` khoảng 27 triệu và `visual.deepstack_merger_list` khoảng 82 triệu. Số tôi đo được là **109.105.152** tham số projector (E2 trừ E1: 142.135.296 − 33.030.144).
- **E1:** chỉ LoRA trên 252 module của LLM (33,0 triệu tham số). ViT và projector đóng băng.
- **E2:** E1 + huấn luyện đầy đủ projector (thêm 109,1 triệu).
- **E3:** E2 + LoRA trên 96 module của ViT (tổng 348 module). Thêm 6.291.456 tham số, tổng 148.426.752.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` của E0 là **0,881**, tức khoảng 37 trên 311 ảnh có output không đọc được thành JSON hợp lệ. Hai lỗi phổ biến nhất:
1. **Bịa và bỏ sót box.** Crack có 243 box thừa (precision 0,036) và F1 chỉ 0,033. Pothole thì bỏ sót nhiều: tìm đúng 10 trên 121 box (recall 0,083). Tính chung E0 chỉ tìm đúng 46 trên 583 box (recall 0,079).
2. **JSON hỏng** ở khoảng 12% ảnh. Nguyên nhân thường gặp: model lặp đi lặp lại rất nhiều box giống nhau (đa số là crack hoặc pothole) cho tới khi hết 256 token, nên JSON bị cắt giữa chừng.

Khi đã khớp được box thì vị trí khá chuẩn (mean IoU 0,596), tức model hiểu thang 0–1000, nhưng không biết ba lớp của bài này.
Ví dụ output thô (box bị lặp hai lần, không có thật): `[{"label": "manhole", "bbox_2d": [309, 684, 399, 714]}, {"label": "manhole", "bbox_2d": [309, 684, 399, 714]}]`.
Ví dụ output hỏng (bị cắt giữa chừng, các box crack nối nhau lặp mãi, mình rút gọn phần đầu): `[{"label": "crack", "bbox_2d": [389, 500, 450, 520]}, {"label": "crack", "bbox_2d": [450, 520, 500, 540]}, {"label": "crack", "bbox_2d": [500, 540, 550, 560]}, ... {"label": "crack", "bbox_2d": [650` (hết 256 token ở đây, thiếu `]`). Trong 60 ảnh val đầu tiên tôi chạy thử có 6 output như vậy.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
F1 của E3 là **0,432** khi chấm đúng thang, và **0,000** khi chấm như thể là pixel. Tôi dùng đúng file dự đoán `e3.jsonl`, chỉ đổi cờ `--coord pixel`. JSON vẫn hợp lệ (valid 1,0) nhưng không có box nào khớp (TP = 0, meanIoU = 0). Lý do: model trả số trên thang 0–1000, còn khi chấm theo pixel, `score.py` hiểu đó là pixel rồi chia cho 640 và 360. Toạ độ bị giãn khoảng 1,6 lần theo chiều ngang và 2,8 lần theo chiều dọc, nên box lệch hẳn và IoU gần 0.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mục tiêu là dạy model **sinh ra JSON**. Prompt lúc nào cũng giống nhau và 220 token ảnh (ảnh 640×360) không phải thứ model cần dự đoán. Nếu tính loss trên chúng, phần lớn loss là các token dễ hoặc không đoán được; gradient bị loãng và model tốn sức học thuộc prompt. Con số loss nhìn đẹp nhưng không phản ánh chất lượng JSON. Trong code, tôi gán `labels = -100` cho phần prompt và bỏ cả ký tự xuống dòng thừa sau `<|im_end|>`, nên chỉ phần JSON và `<|im_end|>` được học. Tôi kiểm tra điều này bằng test trong `my_tests/test_training.py`. Sau huấn luyện, `parse_valid_rate` của E1 là 1,0 (E0 là 0,881).

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Khi bị lượng tử hoá 4-bit, trọng số được nén thành các khối số nguyên nhỏ, không phải số thực để cập nhật. Chỉ có thể gắn LoRA lên trên, không huấn luyện đầy đủ được. **Tôi đã gặp lỗi này thật:** dù đã ghi `llm_int8_skip_modules=["visual", "lm_head"]`, ViT vẫn bị nén. Dấu hiệu là số tham số projector ra 54.579.200 (khoảng một nửa của 109 triệu) và huấn luyện báo lỗi gradient fp16. Cách xử lý: thêm `"model.visual"` vào danh sách bỏ qua, và thêm kiểm tra trong `load_model` để dừng ngay nếu còn lớp 4-bit trong ViT hoặc `lm_head`. Sau khi sửa, projector là 109.105.152 tham số đúng như dự kiến. Projector phải ở float32 vì fp16 có độ chính xác thấp và dễ underflow gradient nhỏ, còn GradScaler không cho phép gradient fp16.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các ảnh là frame cắt từ video, nên hai frame liền kề gần như giống hệt nhau. Chia ngẫu nhiên sẽ đặt ảnh gần giống nhau ở cả train và test, model "nhớ" ảnh train và được chấm điểm cao một cách giả tạo. Chia theo phiên quay thì cả phiên nằm trong một split, nên test là cảnh mới thật sự. (Tôi chưa chạy thử chia ngẫu nhiên để đo mức chênh.)

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **E1 → E2:** F1 tăng 0,252 → 0,325 (+0,073), nhưng hai khoảng tin cậy chỉ chạm nhau một chút (0,289 so với 0,286), nên chưa chắc chắn. Giá: VRAM +1,0 GB (10,475 → 11,469), thời gian gần như bằng nhau (45,16 → 45,61 phút).
- **E2 → E3:** F1 tăng 0,325 → 0,432 (+0,107), hai khoảng tin cậy **không chạm nhau**, nên đáng tin. Giá: VRAM +0,18 GB, thời gian +6 phút, thêm 6,3 triệu tham số.

**E3 đáng đồng tiền nhất.** Nó cũng giảm box thừa (FP/ảnh 1,21 → 1,04 → 0,79). Mở projector (E2) chỉ có lợi nhẹ so với E1, phải thêm LoRA trên ViT thì mới tăng rõ.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 của E3 là 0,390–0,475, rộng khoảng 0,085 (±0,04). Tập test có 311 ảnh và 583 box, nên chênh lệch nhỏ hơn khoảng 0,08 thì chưa kết luận được. Ví dụ E1 so với E2 chênh 0,073 và hai CI chạm nhau, nên không nói chắc E2 tốt hơn E1. Còn E3 so với E2, hay E4 (10%) so với E3, chênh lệch lớn hơn nhiều và CI không chạm nhau, nên tin được. Ngoài ra, tôi chỉ chạy một lần mỗi setup (một seed), và bootstrap lấy mẫu theo ảnh trong khi các ảnh cùng phiên giống nhau, nên khoảng tin cậy thực tế có thể còn rộng hơn con số này.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Với E3: **pothole kém nhất** (F1 0,302; bỏ sót 90 trên 121 box), rồi crack (0,371; bỏ sót 194 và thừa 162 box). Manhole tốt nhất (0,646). Không phải do thiếu mẫu: train có 1.074 pothole nhiều hơn 691 manhole mà manhole vẫn tốt hơn. Nguyên nhân hợp lý hơn là hình dạng và kích thước: manhole tròn, rõ, đứng riêng; pothole và crack nhỏ (box trung vị chỉ khoảng 0,7% khung hình), viền mờ và hay nằm sát nhau. Ảnh `report/evidence/gallery_e3.png` (xanh = box đúng, đỏ = dự đoán): hàng trên là ba ảnh chỉ có một nắp cống, model đoán đúng; hàng dưới là ba cảnh nhiều hư hại (5, 5, 6 box thật) mà model chỉ đoán được 2, 2, 3 box: bỏ sót các vết nhỏ và gộp vết nứt dài thành một box lớn.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi đổi **`data_size`**: 10% so với 100% ảnh train, trên nền E3, mọi thứ khác giữ nguyên (300 bước, cùng lr). F1 giảm từ **0,432 xuống 0,263**, FP/ảnh tăng 0,79 → 1,73, và `parse_valid_rate` hơi thấp hơn (0,997). Hai CI95 là 0,390–0,475 và 0,230–0,299, **không chạm nhau**, nên kết luận "ít dữ liệu hơn thì kém hơn" là đáng tin. Một điều cần lưu ý: vì giữ cố định 300 bước, bản 10% (khoảng 142 ảnh) học lặp lại ảnh nhiều hơn nhiều (khoảng 17 vòng so với 1,7 vòng), nên kết quả có thể do cả ít dữ liệu lẫn overfit. Giá trị 100% chính là kết quả E3, tôi dùng lại chứ không chạy thêm.

## Bonus E5 — GRPO với reward IoU

**Hàm reward** (`src/rewards.py`): JSON hỏng = 0; cả hai rỗng = 1; còn lại `0,2 + 0,8 × tổng IoU các cặp khớp / max(số box thật, số box đoán, 1)`. Box bịa bị phạt qua mẫu số. Có 10 test (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng, IoU thấp...), chạy không cần GPU.

**Cách chạy** (`src/grpo.py`): khởi tạo từ adapter E3. Mỗi bước lấy 2 ảnh, sinh 4 câu trả lời cho mỗi ảnh, chấm reward, rồi tăng xác suất những câu tốt hơn trung bình nhóm (advantage = reward trừ trung bình nhóm, chia độ lệch chuẩn). 60 bước, tối đa 128 token, lr 5e-6, không KL (beta = 0). Tôi tự viết vòng lặp thay vì dùng `trl.GRPOTrainer` để dùng lại phần nạp adapter và projector đã chạy được, và để dễ gỡ lỗi. Tốn 32,6 phút, VRAM đỉnh 6,3 GB. Log từng bước nằm ở `work/e5/run.json` (không commit).

**Kết quả so với SFT (E3), trên 311 ảnh test:**

| | F1 | CI95 | Precision | Recall | FP/ảnh | F1 manhole |
|---|---|---|---|---|---|---|
| E3 (SFT) | 0,432 | 0,390–0,475 | 0,482 | 0,391 | 0,79 | 0,646 |
| E5 (GRPO) | 0,439 | 0,395–0,480 | 0,528 | 0,376 | 0,63 | 0,674 |

F1 chỉ tăng 0,007 và hai khoảng tin cậy chồng lên nhau gần hết, nên **không thể nói GRPO tốt hơn SFT**. Cái thay đổi là sự đánh đổi: ít box thừa hơn (precision tăng) nhưng bỏ sót nhiều hơn một chút (recall giảm).

**Reward trong lúc huấn luyện** gần như đi ngang: trung bình từng phần tư là 0,418 → 0,389 → 0,411 → 0,408. Lý do có thể là chỉ 60 bước, lr nhỏ và mỗi bước chỉ 2 ảnh nên điểm dao động theo ảnh được chọn. Có 2 trong 60 bước cả 8 câu cùng điểm nên không học được gì, và số câu có tín hiệu giảm dần (trung bình 7,7 câu/bước ở phần tư đầu, 5,9 ở phần tư cuối).

**Reward hacking:** không thấy dấu hiệu nặng. Tỉ lệ trả `[]` là 0 cả khi huấn luyện lẫn trên test, và không có box nào lớn hơn 30% khung hình. Có một lệch nhẹ về phía "thận trọng": số box dự đoán trên test giảm từ 473 xuống 415 (trung bình 1,52 → 1,33 box/ảnh, trong khi số thật là 1,87), chủ yếu là crack (267 → 221). Box cũng hơi to hơn (trung vị 1,1% → 1,3% khung hình; số box lớn hơn 10% khung hình tăng 16 → 26). Điều này phù hợp với việc mẫu số `max(n_gt, n_pred)` phạt box thừa, nhưng tôi chỉ có một lần chạy nên chưa chắc chắn nguyên nhân.

## Nhìn lại

- Tôi gặp lỗi mà công cụ chấm không bắt được: ViT bị lượng tử hoá 4-bit dù đã ghi bỏ qua. Chỉ nhờ số tham số projector lệch (54,6 triệu thay vì 109 triệu) và lỗi gradient fp16 mới phát hiện ra. Sau đó tôi thêm kiểm tra tự động trong `load_model`.
- Tôi chưa hiểu rõ vì sao riêng các bias của projector vẫn là fp16 khi đổi `.data` (nguyên nhân gốc là lượng tử hoá, nhưng cơ chế chi tiết tôi chưa tìm hiểu).
- Thời gian train thật khoảng 45–52 phút mỗi lần, lâu hơn ước tính 20–30 phút của đề.
- E4 chưa tách được ảnh hưởng của "ít dữ liệu" và "nhiều vòng lặp"; muốn rõ hơn cần giữ số vòng lặp (epoch) cố định thay vì số bước.
- Tôi chỉ dùng một seed và một lần chạy mỗi setup, nên chưa biết độ dao động giữa các lần huấn luyện.
