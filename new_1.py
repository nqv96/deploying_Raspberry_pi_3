import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
import argparse
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
    """
    Tính toán IoU giữa hai bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Tính tọa độ giao điểm
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1]) 
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Tính diện tích giao nhau
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Tính diện tích của hai boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Tính IoU
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-7)
    return iou

def merge_boxes(boxes, scores, box_idx, overlap_indices):
    """
    Merge boxes với trọng số theo confidence score
    """
    if not overlap_indices.size:
        return boxes[box_idx]
    
    # Thêm box hiện tại vào danh sách boxes cần merge
    merge_indices = np.append(overlap_indices, box_idx)
    
    # Tính toán box mới bằng trọng số
    weights = scores[merge_indices, np.newaxis]
    weighted_boxes = boxes[merge_indices] * weights
    merged_box = weighted_boxes.sum(axis=0) / weights.sum()
    
    return merged_box

def custom_nms(boxes, scores, iou_threshold=0.45, merge=True):
    """
    NMS tùy chỉnh với tùy chọn merge boxes
    Tương tự chức năng MergeNMS
    """
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    # Sắp xếp boxes theo score (từ cao xuống thấp)
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    
    keep_boxes = []
    keep_scores = []
    
    while len(boxes) > 0:
        # Lấy box có score cao nhất
        current_box = boxes[0]
        current_score = scores[0]
        
        # Thêm vào danh sách kết quả
        if merge:
            # Tính IoU với tất cả boxes còn lại
            ious = np.array([calculate_iou(current_box, box) for box in boxes[1:]])
            # Tìm các boxes có IoU > threshold
            overlap_indices = np.where(ious > iou_threshold)[0] + 1  # +1 vì bỏ qua index 0
            
            # Merge boxes nếu có overlapping
            merged_box = merge_boxes(boxes, scores, 0, overlap_indices)
            keep_boxes.append(merged_box)
        else:
            keep_boxes.append(current_box)
        
        keep_scores.append(current_score)
        
        # Loại bỏ box vừa xử lý
        if merge:
            # Loại bỏ cả các boxes bị overlapping
            overlap_mask = np.ones(len(boxes), dtype=bool)
            overlap_mask[0] = False  # Box hiện tại
            
            if len(boxes) > 1:
                ious = np.array([calculate_iou(current_box, box) for box in boxes[1:]])
                overlap_indices = np.where(ious > iou_threshold)[0] + 1
                overlap_mask[overlap_indices] = False
            
            boxes = boxes[overlap_mask]
            scores = scores[overlap_mask]
        else:
            # Chỉ loại bỏ box hiện tại và các boxes bị overlap
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

def draw_boxes(img, boxes, scores):
    draw = ImageDraw.Draw(img)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill="blue")
        draw.text((x1, y1), f"person {score:.2f}", fill="red")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# def main():

def main():
    parser = argparse.ArgumentParser(description="MobileNetV2+YOLOv3 TinyLite human detection script")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--images", nargs='+', required=True, help="List of image paths to test")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default: 0.1)")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold (default: 0.45)")
    parser.add_argument("--output", help="Directory to save output images (optional)")
    parser.add_argument("--show", action="store_true", help="Display detection results")
    
    args = parser.parse_args()
    
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]

    if args.show and len(args.images) > 1:
        plt.figure(figsize=(5*min(len(args.images), 3), 5*((len(args.images)-1)//3+1)))
    
    for i, image_path in enumerate(args.images):
        # print(f"\nProcessing image: {image_path}")
        
        # Load và resize ảnh
        image = Image.open(image_path).convert("RGB")
        image = image.resize(resolution[::-1])  # Resize theo width, height
        image_np = preprocess_image(image)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], image_np)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        # print(f"Inference time: {inference_time*1000:.1f} ms")
        
        # Decode all outputs
        all_boxes = []
        all_scores = []
        for j, detail in enumerate(output_details):
            output = interpreter.get_tensor(detail['index'])
            output = np.squeeze(output)
            
            # Reshape output
            output = output.reshape((output.shape[0], output.shape[1], 3, 6))
            
            boxes, scores = decode_yolo_output(output, anchors[j], strides[j])
            mask = scores > args.conf
            if np.any(mask):
                all_boxes.append(boxes[mask])
                all_scores.append(scores[mask])

        # Kết hợp boxes từ các scales khác nhau
        if all_boxes and any(len(b) > 0 for b in all_boxes):
            all_boxes = np.concatenate([b for b in all_boxes if len(b) > 0], axis=0)
            all_scores = np.concatenate([s for s in all_scores if len(s) > 0], axis=0)
            
            # print(f"Total boxes before NMS: {len(all_boxes)}")
            
            # Áp dụng custom NMS với merge=True (giống MergeNMS)
            start_time = time.time()
            final_boxes, final_scores = custom_nms(all_boxes, all_scores, iou_threshold=args.nms, merge=True)
            nms_time = time.time() - start_time
            # print(f"NMS time: {nms_time*1000:.1f} ms")
            
            # print(f"Final boxes after NMS: {len(final_boxes)}")
            
            # Load ảnh gốc
            image_raw = Image.open(image_path).convert("RGB")
            w_orig, h_orig = image_raw.size

            # Scale boxes về kích thước ảnh gốc
            scale_w = w_orig / resolution[1]
            scale_h = h_orig / resolution[0]

            final_boxes[:, [0, 2]] *= scale_w
            final_boxes[:, [1, 3]] *= scale_h
            
            # Clip các giá trị
            final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, w_orig)
            final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, h_orig)
            
            # In thông tin các box
            for k, (box, score) in enumerate(zip(final_boxes, final_scores)):
                x1, y1, x2, y2 = box
                print(f"[{k}] x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, score={score:.2f}")
            
            # Lưu hoặc hiển thị kết quả
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                output_path = os.path.join(args.output, os.path.basename(image_path))
                draw = ImageDraw.Draw(image_raw)
                for box, score in zip(final_boxes, final_scores):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill="blue")
                    draw.text((x1, y1), f"person {score:.2f}", fill="red")
                image_raw.save(output_path)
                print(f"Result saved to {output_path}")
            
            # Hiển thị kết quả nếu --show được chỉ định
            if args.show:
                if len(args.images) > 1:
                    plt.subplot(((len(args.images)-1)//3)+1, min(len(args.images), 3), i+1)
                    plt.title(os.path.basename(image_path))
                    # Sử dụng hàm draw_boxes đã được định nghĩa trước đó
                    # Không hiển thị ngay mà đợi tất cả ảnh xử lý xong
                    draw = ImageDraw.Draw(image_raw)
                    for box, score in zip(final_boxes, final_scores):
                        x1, y1, x2, y2 = box
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                        draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill="blue")
                        draw.text((x1, y1), f"person {score:.2f}", fill="red")
                    plt.imshow(image_raw)
                    plt.axis('off')
                else:
                    # Nếu chỉ có một ảnh, sử dụng trực tiếp hàm draw_boxes
                    draw_boxes(image_raw, final_boxes, final_scores)
        else:
            print("No detections found above confidence threshold.")
    
    # Hiển thị tất cả ảnh nếu --show được chỉ định và có nhiều ảnh
    if args.show and len(args.images) > 1:
        plt.tight_layout()
        plt.show()
    parser = argparse.ArgumentParser(description="MobileNetV2+YOLOv3 TinyLite human detection script")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--image", required=True, help="Path to image for detection")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default: 0.1)")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold (default: 0.45)")
    parser.add_argument("--output", help="Path to save output image (optional)")
    parser.add_argument('--show', action='store_true', help="Show result images using matplotlib")
    
    args = parser.parse_args()
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]

    # Load và resize ảnh
    image = Image.open(args.image).convert("RGB")
    image = image.resize(resolution[::-1])  # Resize theo width, height
    image_np = preprocess_image(image)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], image_np)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Decode all outputs
    all_boxes = []
    all_scores = []
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])
        output = np.squeeze(output)
        
        # Reshape output
        output = output.reshape((output.shape[0], output.shape[1], 3, 6))
        
        boxes, scores = decode_yolo_output(output, anchors[i], strides[i])
        mask = scores > args.conf
        if np.any(mask):
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])

    # Kết hợp boxes từ các scales khác nhau
    if all_boxes and any(len(b) > 0 for b in all_boxes):
        all_boxes = np.concatenate([b for b in all_boxes if len(b) > 0], axis=0)
        all_scores = np.concatenate([s for s in all_scores if len(s) > 0], axis=0)
        
        # print(f"Total boxes before NMS: {len(all_boxes)}")
        
        # Áp dụng custom NMS với merge=True (giống MergeNMS)
        start_time = time.time()
        final_boxes, final_scores = custom_nms(all_boxes, all_scores, iou_threshold=nms_threshold, merge=True)
        nms_time = time.time() - start_time
        # print(f"NMS time: {nms_time*1000:.1f} ms")
        
        # print(f"Final boxes after NMS: {len(final_boxes)}")
        
        # Load ảnh gốc
        image_raw = Image.open(args.image).convert("RGB")
        w_orig, h_orig = image_raw.size

        # Scale boxes về kích thước ảnh gốc
        scale_w = w_orig / resolution[1]
        scale_h = h_orig / resolution[0]

        final_boxes[:, [0, 2]] *= scale_w
        final_boxes[:, [1, 3]] *= scale_h
        
        # Clip các giá trị
        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, w_orig)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, h_orig)
        
        if args.show:
            plt.figure(figsize=(15, 5))
        draw = ImageDraw.Draw(image_raw)
        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.ellipse([(x1 + x2)/2 - 2, (y1 + y2)/2 - 2, (x1 + x2)/2 + 2, (y1 + y2)/2 + 2], fill="blue")
            draw.text((x1, y1), f"person {score:.2f}", fill="red")
        
        # Lưu hoặc hiển thị kết quả
        if args.output:
            image_raw.save(args.output)
            print(f"Result saved to {args.output}")
        else:
            plt.figure(figsize=(8, 8))
            plt.imshow(image_raw)
            plt.axis("off")
            plt.show()
        
        for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
            x1, y1, x2, y2 = box
            print(f"[{i}] x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, score={score:.2f}")
    else:
        print("No detections found above confidence threshold.")
    
if __name__ == "__main__":
    main()