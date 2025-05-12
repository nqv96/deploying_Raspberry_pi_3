import cv2
import numpy as np
import time
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from tflite_runtime.interpreter import Interpreter

# -----------------------------
# Step 1: Load the TFLite model
# -----------------------------
def load_model(model_path: str) -> Interpreter:
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# -----------------------------
# Step 2: Prepare input image
# -----------------------------
def preprocess_image(img: np.ndarray, input_shape: tuple) -> np.ndarray:
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to model's expected size
    img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[2]))
    # Normalize to 0-1
    img_norm = img_resized.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(img_norm, axis=0)

# -----------------------------
# Step 3: Run inference
# -----------------------------
def run_inference(interpreter: Interpreter, input_data: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])  # shape: [1, N, 85]
    return preds

# -----------------------------
# Step 4: Post-process detections
# -----------------------------
def postprocess(predictions: np.ndarray, orig_img: np.ndarray,
                conf_threshold=0.3, iou_threshold=0.45):
    # predictions: [1, num_boxes, 85]
    pred = predictions[0]
    # Boxes: xywh normalized (0-1)
    boxes = pred[:, :4]
    objectness = pred[:, 4:5]
    class_scores = pred[:, 5:]
    # Multiply objectness with class scores
    scores = objectness * class_scores
    # For each box, get best class
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    # Convert to pixel coordinates
    h, w = orig_img.shape[:2]
    # xywh to xyxy
    boxes_xyxy = []
    for x, y, bw, bh in boxes:
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        boxes_xyxy.append([x1, y1, x2 - x1, y2 - y1])  # for NMSBoxes: x, y, w, h
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy, confidences.tolist(), conf_threshold, iou_threshold)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes_xyxy[i]
            results.append((class_ids[i], float(confidences[i]), (x, y, x + w_box, y + h_box)))
    return results

# -----------------------------
# Step 5: Draw boxes
# -----------------------------
def draw_detections(img: np.ndarray, detections: list, class_names: list):
    for cls_id, score, (x1, y1, x2, y2) in detections:
        label = f"{class_names[cls_id]}: {score:.2f}"
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw center point
        cv2.circle(img, (int((x1 + x2)/2), int((y1 + y2)/2)), 2, (255, 0, 0), -1)
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img

# -----------------------------
# Utility functions for UI
# -----------------------------
def get_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def get_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"{memory_info.rss / (1024 * 1024):.2f} MB"

def process_frame(frame, interpreter, input_details, output_details, resolution, conf_threshold=0.3, iou_threshold=0.45, class_names=None):
    # Preprocess
    input_tensor = preprocess_image(frame, input_details[0]['shape'])
    # Inference
    preds = run_inference(interpreter, input_tensor)
    # Post-process
    dets = postprocess(preds, frame, conf_threshold, iou_threshold)
    
    # Extract boxes and scores for returning
    boxes = []
    scores = []
    for cls_id, score, (x1, y1, x2, y2) in dets:
        # Check if we want only people (class_id 0)
        if class_names is None or class_names[cls_id] == 'person':
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    # Draw if class_names is provided
    if class_names:
        frame = draw_detections(frame, dets, class_names)
    
    return boxes, scores

# -----------------------------
# Main function
# -----------------------------
def main():
    # 80 COCO classes
    CLASS_NAMES = [
        'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
        'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed',
        'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave',
        'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]
    
    parser = argparse.ArgumentParser(description="Real-time object detection using YOLOv5n TFLite model")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold (default: 0.45)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    
    args = parser.parse_args()
    
    # Load model
    interpreter = load_model(args.model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if hasattr(cv2, 'CAP_V4L2'):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Initialize Tkinter
    root = tk.Tk()
    root.title("YOLOv5n Object Detection")
    
    # Create frame for video and information
    main_frame = tk.Frame(root)
    main_frame.pack(padx=10, pady=10)
    
    # Create label for video display
    label = tk.Label(main_frame)
    label.pack()
    
    # Create frame for detection information
    info_frame = tk.Frame(main_frame)
    info_frame.pack(fill=tk.X, pady=5)
    
    # Label for detected objects count
    detection_label = tk.Label(info_frame, text="Detected objects: 0", font=("Arial", 12))
    detection_label.pack(side=tk.LEFT, padx=5)
    
    # Label for FPS
    fps_label = tk.Label(info_frame, text="FPS: 0", font=("Arial", 12))
    fps_label.pack(side=tk.RIGHT, padx=5)
    
    # Create frame for system information
    system_frame = tk.Frame(main_frame)
    system_frame.pack(fill=tk.X, pady=5)
    
    # Label for model size
    model_size = get_model_size(args.model)
    model_label = tk.Label(system_frame, text=f"Model size: {model_size}", font=("Arial", 10))
    model_label.pack(side=tk.LEFT, padx=5)
    
    # Label for memory usage
    memory_label = tk.Label(system_frame, text="Memory: 0 MB", font=("Arial", 10))
    memory_label.pack(side=tk.RIGHT, padx=5)
    
    # Create frame for model information
    model_info_frame = tk.Frame(main_frame)
    model_info_frame.pack(fill=tk.X, pady=5)
    
    # Label for model resolution
    resolution_label = tk.Label(model_info_frame, text=f"Input resolution: {resolution[0]}x{resolution[1]}", font=("Arial", 10))
    resolution_label.pack(side=tk.LEFT, padx=5)
    
    # Label for thresholds
    threshold_label = tk.Label(model_info_frame, text=f"Conf: {args.conf}, NMS: {args.nms}", font=("Arial", 10))
    threshold_label.pack(side=tk.RIGHT, padx=5)
    
    # Create exit button
    exit_button = tk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    
    def update_frame():
        nonlocal frame_count, start_time
        
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera")
            root.quit()
            return
        
        # Process frame
        boxes, scores = process_frame(frame, interpreter, input_details, output_details, 
                                    resolution, args.conf, args.nms, CLASS_NAMES)
        
        # Count valid detections
        valid_detections = len(boxes)
        
        # Update detection information
        detection_label.config(text=f"Detected objects: {valid_detections}")
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            fps_label.config(text=f"FPS: {fps:.1f}")
            frame_count = 0
            start_time = time.time()
            
            # Update memory usage every second
            memory_label.config(text=f"Memory: {get_memory_usage()}")
        
        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        
        # Convert to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        label.imgtk = imgtk  # Keep reference to prevent garbage collection
        label.configure(image=imgtk)
        
        # Schedule next frame update
        root.after(10, update_frame)  # Update every 10ms
    
    print("Starting object detection...")
    
    # Start frame updates
    update_frame()
    
    # Run main loop
    root.mainloop()
    
    # Cleanup
    cap.release()

if __name__ == "__main__":
    main()