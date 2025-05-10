import cv2
# print(cv2.getBuildInformation())
# Mở webcam (0 là camera mặc định của laptop)
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
# print("Success?", ret)
# cap.release()
# Different backends 
# backends = [
#     cv2.CAP_DSHOW,    # Windows
#     cv2.CAP_V4L2,     # Linux
#     cv2.CAP_MSMF,     # Microsoft Media Foundation
#     cv2.CAP_ANY       # Auto-detect
# ]
# Chờ camera khởi động
if not cap.isOpened():
    print("Không mở được camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print('Khong mo duoc')
        break
    cv2.imshow("fram", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()