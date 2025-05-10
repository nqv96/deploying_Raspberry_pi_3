import cv2

# Mở camera (thường là /dev/video0 tương ứng với index 0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # V4L2 cho Raspberry Pi

# Nếu cần, có thể đặt độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # hoặc 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # hoặc 720

# Kiểm tra camera có mở được không
if not cap.isOpened():
    print("Không mở được camera.")
    exit()

print("Camera đã mở thành công. Nhấn 'q' để thoát.")

# Vòng lặp hiển thị video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame.")
        break

    cv2.imshow("Camera", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()