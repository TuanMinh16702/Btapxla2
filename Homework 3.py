import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đọc hình ảnh nhị phân "actontBin.bin"

image_data = np.fromfile('binfile/actontBin.bin', dtype=np.uint8, count=256*256)
hinh_anh_nhap = image_data.reshape(256, 256)

# Thiết kế mẫu "T"
template = np.array([[1, 1, 1],
                    [0, 1, 0],
                    [0, 1, 0]], dtype=np.uint8)

# Tính kích thước mẫu
chieu_cao_mau, chieu_rong_mau = template.shape

# Khởi tạo hình ảnh kết quả J
hinh_anh_ket_qua = np.zeros_like(hinh_anh_nhap, dtype=np.float32)

# Lặp qua hình ảnh đầu vào và tính giá trị độ tương quan tại mỗi điểm ảnh sử dụng mẫu
for y in range(chieu_cao_mau // 2, 256 - chieu_cao_mau // 2):
    for x in range(chieu_rong_mau // 2, 256 - chieu_rong_mau // 2):
        khu_vuc_xung_quanh = hinh_anh_nhap[y - chieu_cao_mau // 2:y + chieu_cao_mau // 2 + 1,
                                          x - chieu_rong_mau // 2:x + chieu_rong_mau // 2 + 1]
        do_tuong_quan = np.sum(khu_vuc_xung_quanh * template)
        hinh_anh_ket_qua[y, x] = do_tuong_quan

# Threshold để tạo hình ảnh nhị phân Jz
nguong = np.max(hinh_anh_ket_qua) * 0.9  # Điều chỉnh ngưỡng theo nhu cầu
Jz = np.where(hinh_anh_ket_qua > nguong, 255, 0).astype(np.uint8)

# Hiển thị cả hình ảnh gốc và hình ảnh kết quả
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(hinh_anh_nhap, cmap='gray')
plt.title("Hình ảnh gốc")

plt.subplot(1, 2, 2)
plt.imshow(Jz, cmap='gray')
plt.title("Hình ảnh sau khi tìm 'T'")

plt.show()