Đây là end-to-end scoring supporter, kết hợp YOLOv9 và Parseq-tiny, cụ thể:
YOLOv9: detect 3 hoặc 5 class (đọc comment trong file e2e_run.py để thay đổi 2 model)
Parseq-tiny: detect trực tiếp ảnh selected_ans CÓ nhiễu khoanh tròn

Code để chạy là file e2e_run.py

Hướng dẫn:
1. Về các model
- Các model của nhóm sẽ được lưu trong folder "pretrained_model"
- Hiện tại, trong pretrained model chỉ có model "Parseq-tiny", là model tốt nhất của nhóm
- Đối với YoloV9, thì cần lên onedrive của nhóm để tải về model tốt nhất
2. Về cài đặt các package
- File requirements.txt hiện tại có thể thiếu một vài library để chạy, nếu thiếu thì tự cài và báo cho Thịnh
- Có thể một số library trong requirements không đúng version, nếu xảy ra lỗi thư viện, liên hệ Thịnh
3. Về cách chạy e2e_run.py
- Hiện code đang đơn giản, chỉ infer cho một ảnh, kết quả trả về là visualization của ảnh đó với Bounding box của Question_id và Selected_ansewer và kết quả Nhận diện bên trên

Nhiệm vụ:
Cần tự code lại file e2e_run để test trên toàn bộ test set. Kết quả trả về cần là một file json, mỗi dòng 
là một dictionary với 3 trường thông tin 
+) "cls": là 1 trong 3/5 class (0,1,2,3,4)
+) "points": tọa độ các điểm x1y1x2y2 của bounding box
+) "text": kết quả nhận diện cho bounding box đó
(Xem ví dụ file test_set/results.json)