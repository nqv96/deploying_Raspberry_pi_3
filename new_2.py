import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw, ImageTk
import time
import argparse
import tkinter as tk
import psutil
import os

# Tham số anchor cho từng scale
anchors = [
    np.array([[116, 90], [156, 198], [373, 326]]),  # scale 32x
    np.array([[30, 61], [62, 45], [59, 119]]),      # scale 16x
    np.array([[10, 13], [16, 30], [33, 23]])        # scale 8x
]
strides = [32, 16, 8]
num_classes = 1
conf_threshold = 0.1
nms_threshold = 0.45
input_size = 416  # Kích thước ảnh đầu vào của mô hình

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_yolo_output(feat, anchors, stride):
    grid_h, grid_w, num_anchors, _ = feat.shape
    
    # Tạo grid cells
    grid_x = np.arange(grid_w)
    grid_y = np.arange(grid_h)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # Reshape để broadcast
    grid_x = grid_x.reshape(grid_h, grid_w, 1, 1)
    grid_y = grid_y.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((grid_x, grid_y), axis=-1)
    
    # Decode box coordinates
    raw_box_centers = feat[..., 0:2]  # x, y
    raw_box_scales = feat[..., 2:4]   # w, h
    
    # Tính toán center coordinates
    box_centers = (sigmoid(raw_box_centers) + grid) * stride
    
    # Reshape anchors
    reshaped_anchors = anchors.reshape(1, 1, num_anchors, 2)
    
    # Tính toán width và height
    box_scales = np.exp(raw_box_scales) * reshaped_anchors
    
    # Tính toán x1y1x2y2 format
    wh = box_scales / 2.0
    box_x1y1 = box_centers - wh
    box_x2y2 = box_centers + wh
    
    # Kết hợp thành boxes
    boxes = np.concatenate([box_x1y1, box_x2y2], axis=-1)
    
    # Tính objectness và class scores
    objectness = sigmoid(feat[..., 4:5])
    class_probs = sigmoid(feat[..., 5:])
    
    # Tính toán final scores
    scores = objectness * class_probs
    
    # Reshape để xử lý dễ hơn
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    
    return boxes, scores

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1]) 
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-7)
    return iou

def merge_boxes(boxes, scores, box_idx, overlap_indices):
    if not overlap_indices.size:
        return boxes[box_idx]
    
    merge_indices = np.append(overlap_indices, box_idx)
    weights = scores[merge_indices, np.newaxis]
    weighted_boxes = boxes[merge_indices] * weights
    merged_box = weighted_boxes.sum(axis=0) / weights.sum()
    
    return merged_box

def custom_nms(boxes, scores, iou_threshold=0.45, merge=True):
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    
    keep_boxes = []
    keep_scores = []
    
    while len(boxes) > 0:
        current_box = boxes[0]
        current_score = scores[0]
        
        if merge:
            ious = np.array([calculate_iou(current_box, box) for box in boxes[1:]])
            overlap_indices = np.where(ious > iou_threshold)[0] + 1
            merged_box = merge_boxes(boxes, scores, 0, overlap_indices)
            keep_boxes.append(merged_box)
        else:
            keep_boxes.append(current_box)
        
        keep_scores.append(current_score)
        
        if merge:
            overlap_mask = np.ones(len(boxes), dtype=bool)
            overlap_mask[0] = False
            
            if len(boxes) > 1:
                ious = np.array([calculate_iou(current_box, box) for box in boxes[1:]])
                overlap_indices = np.where(ious > iou_threshold)[0] + 1
                overlap_mask[overlap_indices] = False
            
            boxes = boxes[overlap_mask]
            scores = scores[overlap_mask]
        else:
            if len(boxes) > 1:
                ious = np.array([calculate_iou(current_box, box) for box in boxes[1:]])
                overlap_mask = ious <= iou_threshold
                boxes = boxes[1:][overlap_mask]
                scores = scores[1:][overlap_mask]
            else:
                break
    
    return np.array(keep_boxes), np.array(keep_scores)

def preprocess_image(image):
    image_np = np.array(image)[None, ...]
    image_np = (image_np / 255) * 2 - 1  # Normalize về [-1, 1]
    return image_np.astype('float32')

def process_frame(frame, interpreter, input_details, output_details, resolution):
    # Chuyển frame từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame về kích thước input của model
    frame_resized = cv2.resize(frame_rgb, resolution[::-1])
    
    # Preprocess
    frame_np = preprocess_image(frame_resized)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], frame_np)
    interpreter.invoke()
    
    # Decode outputs
    all_boxes = []
    all_scores = []
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])
        output = np.squeeze(output)
        output = output.reshape((output.shape[0], output.shape[1], 3, 6))
        
        boxes, scores = decode_yolo_output(output, anchors[i], strides[i])
        mask = scores > conf_threshold
        if np.any(mask):
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
    
    # Kết hợp và xử lý kết quả
    if all_boxes and any(len(b) > 0 for b in all_boxes):
        all_boxes = np.concatenate([b for b in all_boxes if len(b) > 0], axis=0)
        all_scores = np.concatenate([s for s in all_scores if len(s) > 0], axis=0)
        
        final_boxes, final_scores = custom_nms(all_boxes, all_scores, iou_threshold=nms_threshold, merge=True)
        
        # Scale boxes về kích thước frame gốc
        h_orig, w_orig = frame.shape[:2]
        scale_w = w_orig / resolution[1]
        scale_h = h_orig / resolution[0]
        
        final_boxes[:, [0, 2]] *= scale_w
        final_boxes[:, [1, 3]] *= scale_h
        
        # Clip các giá trị
        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, w_orig)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, h_orig)
        
        return final_boxes, final_scores
    
    return np.array([]), np.array([])

def get_model_size(model_path):
    """Lấy kích thước file model theo MB"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"

def get_memory_usage():
    """Lấy thông tin sử dụng bộ nhớ của process hiện tại"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return f"{memory_mb:.1f} MB"

# def calculate_model_metrics(interpreter):
#     """Tính toán FLOPS và số lượng tham số của mô hình chính xác"""
#     total_params = 0
#     total_flops = 0
    
#     # Lấy thông tin về các tensor
#     tensor_details = interpreter.get_tensor_details()
    
#     # Danh sách các layer có tham số có thể huấn luyện được
#     param_layers = ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']
    
#     # Lấy các chi tiết của mô hình
#     try:
#         # Lấy thông tin về graph của mô hình (nếu có)
#         # (TFLite không phải lúc nào cũng cho phép truy cập vào các thông tin này)
#         graph_info = interpreter._get_ops_details()
        
#         for op_info in graph_info:
#             op_code = op_info['op_code']
            
#             if op_code in param_layers:
#                 # Lấy các tensor input/output
#                 inputs = op_info['inputs']
#                 outputs = op_info['outputs']
                
#                 # Lấy tensor weights và biases (thường là input thứ 1 và 2)
#                 if len(inputs) > 1:  # Phải có ít nhất 2 inputs (input và weights)
#                     weight_tensor_idx = inputs[1]
                    
#                     # Lấy thông tin weight tensor
#                     for tensor in tensor_details:
#                         if tensor['index'] == weight_tensor_idx:
#                             # Lấy shape từ tensor info, không load giá trị thực
#                             shape = tensor['shape']
#                             weight_params = np.prod(shape)
#                             total_params += weight_params
                            
#                             # Kiểm tra nếu có bias (thường là input thứ 3)
#                             if len(inputs) > 2:
#                                 bias_tensor_idx = inputs[2]
#                                 for bias_tensor in tensor_details:
#                                     if bias_tensor['index'] == bias_tensor_idx:
#                                         bias_shape = bias_tensor['shape']
#                                         bias_params = np.prod(bias_shape)
#                                         total_params += bias_params
#                                         break
                            
#                             # Tính FLOPS
#                             if op_code == 'CONV_2D':
#                                 # Lấy kích thước đầu vào
#                                 for in_tensor in tensor_details:
#                                     if in_tensor['index'] == inputs[0]:
#                                         input_shape = in_tensor['shape']
#                                         break
                                
#                                 # Lấy kích thước đầu ra
#                                 for out_tensor in tensor_details:
#                                     if out_tensor['index'] == outputs[0]:
#                                         output_shape = out_tensor['shape']
#                                         break
                                        
#                                 # Nếu có đủ thông tin, tính FLOPS cho Conv2D
#                                 if 'input_shape' in locals() and 'output_shape' in locals():
#                                     # Conv FLOPS = 2 * H_out * W_out * C_out * (C_in * K_h * K_w)
#                                     # Trong đó:
#                                     # - H_out, W_out: chiều cao, rộng của output
#                                     # - C_out: số kênh output
#                                     # - C_in: số kênh input
#                                     # - K_h, K_w: chiều cao, rộng của kernel
                                    
#                                     # Giả sử shape của weight là [out_channels, in_channels, kernel_h, kernel_w]
#                                     out_channels = shape[0]
#                                     in_channels = shape[1]
#                                     kernel_h, kernel_w = shape[2], shape[3]
                                    
#                                     # Output thường có shape [batch, height, width, channels]
#                                     out_h, out_w = output_shape[1:3] if len(output_shape) >= 4 else (1, 1)
                                    
#                                     flops = 2 * out_h * out_w * out_channels * in_channels * kernel_h * kernel_w
#                                     total_flops += flops
                                
#                             elif op_code == 'DEPTHWISE_CONV_2D':
#                                 # DEPTHWISE_CONV_2D FLOPS = 2 * H_out * W_out * C_in * K_h * K_w
#                                 # Shape của weight thường là [1, in_channels, kernel_h, kernel_w]
#                                 for out_tensor in tensor_details:
#                                     if out_tensor['index'] == outputs[0]:
#                                         output_shape = out_tensor['shape']
#                                         break
                                
#                                 if 'output_shape' in locals():
#                                     in_channels = shape[1]
#                                     kernel_h, kernel_w = shape[2], shape[3]
                                    
#                                     out_h, out_w = output_shape[1:3] if len(output_shape) >= 4 else (1, 1)
                                    
#                                     flops = 2 * out_h * out_w * in_channels * kernel_h * kernel_w
#                                     total_flops += flops
                                
#                             elif op_code == 'FULLY_CONNECTED':
#                                 # FC FLOPS = 2 * output_size * input_size
#                                 # Weight shape thường là [output_size, input_size]
#                                 output_size, input_size = shape
#                                 flops = 2 * output_size * input_size
#                                 total_flops += flops
                            
#                             break
#     except:
#         # Nếu không thể truy cập thông tin chi tiết, sử dụng phương pháp dự phòng
#         for tensor in tensor_details:
#             name = tensor.get('name', '').lower()
            
#             # Chỉ tính weight parameters
#             if ('weight' in name or 'kernel' in name or 'filter' in name) and tensor.get('is_variable', True):
#                 shape = tensor['shape']
#                 params = np.prod(shape)
#                 total_params += params
                
#             # Tính bias parameters
#             elif 'bias' in name and tensor.get('is_variable', True):
#                 shape = tensor['shape']
#                 params = np.prod(shape)
#                 total_params += params
    
#     # Nếu không tính được params bằng cả hai cách, dùng phương pháp ước tính thay thế
#     if total_params == 0:
#         # Ước tính dựa trên kích thước model
#         model_size_bytes = interpreter._model_size  # Có thể không hoạt động với mọi version
#         # Giả sử mỗi tham số chiếm khoảng 4 bytes (float32)
#         total_params = model_size_bytes / 4
    
#     # Chuyển đổi FLOPS sang GigaFLOPS
#     gigaflops = total_flops / (1e9)  # 1 gigaflop = 10^9 flops
    
#     return {
#         'params': f"{int(total_params):,}",
#         'gflops': f"{gigaflops:.2f}"
#     }

def main():
    parser = argparse.ArgumentParser(description="Real-time human detection using TFLite model")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (default: 0.2)")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold (default: 0.45)")
    
    args = parser.parse_args()
    
    # Khởi tạo model
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]
    
    # Tính toán metrics của model
    # model_metrics = calculate_model_metrics(interpreter)
    
    # Khởi tạo camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cv2.CAP_PROP_BUFFERSIZE = 1
    if not cap.isOpened():
        print("Không thể mở camera")
        return
    
    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Khởi tạo Tkinter
    root = tk.Tk()
    root.title("Human Detection")
    
    # Tạo frame chứa video và thông tin
    main_frame = tk.Frame(root)
    main_frame.pack(padx=10, pady=10)
    
    # Tạo label để hiển thị video
    label = tk.Label(main_frame)
    label.pack()
    
    # Tạo frame chứa thông tin detection
    info_frame = tk.Frame(main_frame)
    info_frame.pack(fill=tk.X, pady=5)
    
    # Label hiển thị số người được phát hiện
    detection_label = tk.Label(info_frame, text="Số người phát hiện: 0", font=("Arial", 12))
    detection_label.pack(side=tk.LEFT, padx=5)
    
    # Label hiển thị FPS
    fps_label = tk.Label(info_frame, text="FPS: 0", font=("Arial", 12))
    fps_label.pack(side=tk.RIGHT, padx=5)
    
    # Tạo frame chứa thông tin hệ thống
    system_frame = tk.Frame(main_frame)
    system_frame.pack(fill=tk.X, pady=5)
    
    # Label hiển thị kích thước model
    model_size = get_model_size(args.model)
    model_label = tk.Label(system_frame, text=f"Model size: {model_size}", font=("Arial", 10))
    model_label.pack(side=tk.LEFT, padx=5)
    
    # Label hiển thị bộ nhớ sử dụng
    memory_label = tk.Label(system_frame, text="Memory: 0 MB", font=("Arial", 10))
    memory_label.pack(side=tk.RIGHT, padx=5)
    
    # Tạo frame chứa thông tin model
    model_info_frame = tk.Frame(main_frame)
    model_info_frame.pack(fill=tk.X, pady=5)
    
    # # Label hiển thị số lượng tham số
    # params_label = tk.Label(model_info_frame, text=f"Parameters: {model_metrics['params']}", font=("Arial", 10))
    # params_label.pack(side=tk.LEFT, padx=5)
    
    # # Label hiển thị FLOPS
    # flops_label = tk.Label(model_info_frame, text=f"FLOPS: {model_metrics['gflops']} G", font=("Arial", 10))
    # flops_label.pack(side=tk.RIGHT, padx=5)
    
    # Tạo nút thoát
    exit_button = tk.Button(root, text="Thoát", command=root.quit)
    exit_button.pack(pady=10)
    
    # Biến để tính FPS
    frame_count = 0
    start_time = time.time()
    
    def update_frame():
        nonlocal frame_count, start_time
        
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera")
            root.quit()
            return
        
        # Xử lý frame
        boxes, scores = process_frame(frame, interpreter, input_details, output_details, resolution)
        
        # Vẽ kết quả chỉ khi confidence vượt ngưỡng
        valid_detections = 0
        for box, score in zip(boxes, scores):
            if score > args.conf:  # Chỉ vẽ khi confidence vượt ngưỡng
                valid_detections += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int((x1 + x2)/2), int((y1 + y2)/2)), 2, (255, 0, 0), -1)
                cv2.putText(frame, f"person {score:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Cập nhật thông tin detection
        detection_label.config(text=f"Số người phát hiện: {valid_detections}")
        
        # Tính FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            fps_label.config(text=f"FPS: {fps:.1f}")
            frame_count = 0
            start_time = time.time()
            
            # Cập nhật thông tin bộ nhớ mỗi giây
            memory_label.config(text=f"Memory: {get_memory_usage()}")
        
        # Chuyển từ BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang định dạng PIL
        img = Image.fromarray(frame_rgb)
        
        # Chuyển sang PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Cập nhật label
        label.imgtk = imgtk  # Giữ tham chiếu để tránh bị xóa bởi garbage collector
        label.configure(image=imgtk)
        
        # Lên lịch cập nhật frame tiếp theo
        root.after(10, update_frame)  # Cập nhật mỗi 10ms
    
    print("Bắt đầu phát hiện người...")
    
    # Bắt đầu cập nhật frame
    update_frame()
    
    # Chạy mainloop
    root.mainloop()
    
    # Dọn dẹp
    cap.release()

if __name__ == "__main__":
    main()