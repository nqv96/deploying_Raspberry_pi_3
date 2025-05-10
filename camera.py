import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
# print(cv2.getBuildInformation())
# Mở webcam (0 là camera mặc định của laptop)
#     cv2.CAP_V4L2,     # Linux

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Chờ camera khởi động

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Chuyển từ BGR (OpenCV) sang RGB (Pillow)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    label.imgtk = imgtk  # giữ tham chiếu tránh bị xóa
    label.configure(image=imgtk)
    root.after(10, update_frame)  # cập nhật mỗi 10ms
    

cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
if not cap.isOpened():
    print("Không mở được camera.")
    exit()

root = tk.Tk()
label = tk.Label(root)
label.pack()

update_frame()
root.mainloop()

cap.release()
# plt.ion()  # Chế độ interactive
# fig, ax = plt.subplots()

# img_plot = None
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print('Khong mo duoc')
#         break
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     if img_plot is None:
#         img_plot = ax.imshow(rgb)
#     else:
#         img_plot.set_data(rgb)

#     plt.pause(0.01)
# cap.release()
# plt.ioff()
# plt.show()

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