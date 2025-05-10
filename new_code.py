import torch
import tflite_runtime.interpreter as tflite
import numpy as np
from det_helper import MergeNMS, Yolo3Output
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import psutil
import time


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
input_size = 416  # Kích thước ảnh đầu vào của mô hình (thường là 416 hoặc 320)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_yolo_output(feat, anchors, stride):
    grid_h, grid_w, num_anchors, _ = feat.shape
    
    # Tạo grid với dimensions chính xác
    grid_y = np.arange(grid_h).reshape(grid_h, 1, 1)  # shape (grid_h, 1, 1)
    grid_x = np.arange(grid_w).reshape(1, grid_w, 1)  # shape (1, grid_w, 1)
    
    # Kết hợp grid x và y
    grid_x = np.tile(grid_x, (grid_h, 1, 1))  # shape (grid_h, grid_w, 1)
    grid_y = np.tile(grid_y, (1, grid_w, 1))  # shape (grid_h, grid_w, 1)
    
    grid = np.concatenate((grid_x, grid_y), axis=-1)  # shape (grid_h, grid_w, 2)
    grid = np.expand_dims(grid, axis=2)  # shape (grid_h, grid_w, 1, 2)
    
    # Decode box coordinates
    # Tính toán tọa độ center (x,y) của box
    box_xy = (sigmoid(feat[..., 0:2]) + grid) * stride
    
    # Tính toán width và height của box
    box_wh = np.exp(feat[..., 2:4]) * anchors
    
    # Convert to x1y1x2y2 format
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    boxes = np.concatenate([box_x1y1, box_x2y2], axis=-1)
    
    # Flatten kết quả
    boxes = boxes.reshape(-1, 4)
    objectness = sigmoid(feat[..., 4]).reshape(-1)
    class_probs = sigmoid(feat[..., 5:]).reshape(-1)
    
    # Tính toán score
    scores = objectness * class_probs
    
    return boxes, scores

def preprocess_image(image):
    image_np = np.array(image)[None, ...]
    image_np = (image_np / 255) * 2 - 1
    return image_np.astype('float32')

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Compute IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]
    return keep

def draw_boxes(img, boxes, scores):
    draw = ImageDraw.Draw(img)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill="blue")
        draw.text((x1, y1), f"person {score:.2f}", fill="red")
    # Hiển thị bằng matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()



def main():
    MODEL_PATH = "/home/vuong/my_project/human-detection/detection.tflite"
    # image_path = "/home/vuong/my_project/human-detection/images_test/art_16.jpg"
    image_path = "/home/vuong/my_project/human-detection/images_test/image.png"
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]

    # Load và resize ảnh
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resolution[::-1])  # Resize theo width, height
    image_np = preprocess_image(image)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()
    
    # Decode all outputs
    all_boxes = []
    all_scores = []
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])  # (1, H, W, 18)
        output = np.squeeze(output)  # (H, W, 18)
        output = output.reshape((output.shape[0], output.shape[1], 3, 6))  # (H, W, 3, 6)
        
        boxes, scores = decode_yolo_output(output, anchors[i], strides[i])
        mask = scores > conf_threshold
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    print(f"Total boxes before NMS: {len(all_boxes)}")
    
    # Apply NMS với ngưỡng IoU thấp hơn
    keep_indices = nms(all_boxes, all_scores, 0.3)  # Giảm ngưỡng NMS
    final_boxes = all_boxes[keep_indices]
    final_scores = all_scores[keep_indices]
    
    print(f"Final boxes after NMS: {len(final_boxes)}")
    
    # Load ảnh gốc
    image_raw = Image.open(image_path).convert("RGB")
    w_orig, h_orig = image_raw.size

    # Scale boxes từ kích thước model về kích thước ảnh gốc
    scale_w = w_orig / resolution[0]
    scale_h = h_orig / resolution[1]

    final_boxes[:, [0, 2]] *= scale_w
    final_boxes[:, [1, 3]] *= scale_h
    
    # Clip các giá trị ra ngoài biên
    final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, w_orig)
    final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, h_orig)
    
    draw_boxes(image_raw, final_boxes, final_scores)
    
    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        x1, y1, x2, y2 = box
        print(f"[{i}] x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, score={score:.2f}")
    
if __name__ == "__main__":
    main()

