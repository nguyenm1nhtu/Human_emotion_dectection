<a name="top"></a>

# **Human Emotion Detection**

<details>
  <summary>📖 Mục Lục</summary>

  1. [Giới Thiệu](#giới-thiệu)
  2. [Tiền Xử Lý](#tiền-xử-lý)
  3. [Trích Xuất Đặc Trưng CNN](#trích-xuất-đặc-trưng-cnn)

</details>

## Giới Thiệu
Phát hiện cảm xúc qua hình ảnh là một lĩnh vực đang phát triển nhanh chóng trong AI, giúp cải thiện các ứng dụng từ chăm sóc sức khỏe đến dịch vụ khách hàng. Đề tài này được chọn nhằm tìm hiểu cách học sâu có thể được sử dụng để phát hiện các cảm xúc thông qua hình ảnh khuôn mặt.

### Mục tiêu
Mục tiêu của dự án là phát triển một hệ thống có khả năng phân loại chính xác các biểu cảm khuôn mặt thành một trong bảy loại cảm xúc: Giận dữ, Ghê tởm, Sợ hãi, Vui vẻ, Buồn bã, Ngạc nhiên và Bình thường. Để thực hiện điều này, dự án sẽ sử dụng các kỹ thuật học máy và học sâu như Convolutional Neural Network (CNN), kết hợp với các phương pháp giảm chiều (PCA, LDA), phân cụm (K-Means, DBScan) và các thuật toán phân loại khác (KNN, SoftMax, SVM) để đánh giá và so sánh hiệu quả.

**Mục tiêu bao gồm:**
- Tiền xử lý dữ liệu: Xử lý và chuẩn hóa dữ liệu hình ảnh để đảm bảo chất lượng dữ liệu đầu vào phù hợp cho mô hình.
- Phân tích và giảm chiều dữ liệu: Áp dụng các phương pháp như PCA và LDA để giảm chiều dữ liệu, phân tích các thành phần chính, và trực quan hóa dữ liệu.
- Phân cụm dữ liệu: Sử dụng các thuật toán phân cụm như K-Means, DBScan để nhóm các hình ảnh có biểu cảm tương đồng.
- Phân loại cảm xúc: Sử dụng 3 phương pháp phân loại, bao gồm K-NN, SoftMax, và SVM, để huấn luyện mô hình trên dữ liệu đã tiền xử lý và giảm chiều, sau đó so sánh kết quả.
- Kiểm tra và đánh giá kết quả: Đánh giá hiệu quả của các mô hình dựa trên các chỉ số như accuracy, precision, recall và F1-score. Kiểm tra hiện tượng quá khớp (overfit) và áp dụng biện pháp hiệu chỉnh (regularization) nếu cần.
- Thực hiện hồi quy: Chuyển bài toán phân loại thành bài toán hồi quy để đánh giá các mô hình dựa trên các giá trị dự đoán và so sánh kết quả trên tập dữ liệu gốc và tập dữ liệu đã giảm chiều.

### Tổng Quan Về Bộ Dữ Liệu
Bộ dữ liệu bao gồm các hình ảnh khuôn mặt có kích thước 48x48 pixel ở dạng ảnh xám. Những hình ảnh này đã được tự động căn chỉnh để đảm bảo khuôn mặt nằm ở trung tâm và chiếm cùng một không gian trong mỗi bức hình. Bộ dữ liệu bao gồm 8.052 hình ảnh, trong đó có 6.330 ảnh trong tập huấn luyện (train) và 1.722 ảnh trong tập kiểm tra (test).

[Quay lại đầu trang](#top)

## Tiền Xử Lý
- Đọc và mô tả dữ liệu:
Tải dữ liệu, kiểm tra cấu trúc và số lượng bản ghi: 5.066 ảnh trong tập huấn luyện, 1.264 ảnh trong tập xác thực, 1.722 ảnh trong tập kiểm tra. Thống kê các giá trị dữ liệu ban đầu.
- Chuyển đổi và chuẩn hóa dữ liệu:
Chuyển nhãn cảm xúc thành dạng one-hot encoding.
Chuẩn hóa các giá trị pixel từ khoảng [0, 255] về [0, 1] để đảm bảo đồng nhất dữ liệu đầu vào cho mô hình.
- Tăng cường dữ liệu:
Áp dụng các kỹ thuật tăng cường dữ liệu như xoay, lật, và dịch chuyển để làm phong phú tập huấn luyện và ngăn ngừa quá khớp.
- Mô tả dữ liệu

[Quay lại đầu trang](#top)

## Trích Xuất Đặc Trưng CNN
Sử dụng CNN để chích xuất các đặc trưng quan trọng từ dữ liệu hình ảnh đã qua tiền xử lý thông qua:
- Convolution
- MaxPooling
- Flatten