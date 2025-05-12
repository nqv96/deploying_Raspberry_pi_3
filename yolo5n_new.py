# YOLOv5n TFLite Object Detection , accept
# -----------------------------------------
# This script loads a YOLOv5n TFLite model, runs inference on an input image,
# post-processes the detections (thresholding + NMS), and draws bounding boxes on the image.

import cv2
import numpy as np
from PIL import Image
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
def preprocess_image(image_path: str, input_shape: tuple) -> np.ndarray:
    # Read with OpenCV, convert BGR to RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to model's expected size
    img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[2]))
    # Normalize to 0-1
    img_norm = img_resized.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(img_norm, axis=0), img

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
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img

# -----------------------------
# Main execution example
# -----------------------------
if __name__ == "__main__":
    MODEL_PATH = "/home/vuong/my_project/human-detection/yolov5n-fp16.tflite"
    IMAGE_PATH = "/home/vuong/my_project/human-detection/images_test/34919511_122594755296479_501094598928498688_n.jpg"  # Replace with your image path
    CLASS_NAMES = [  # 80 COCO classes
        'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
        'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed',
        'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave',
        'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]

    # Load model
    interpreter = load_model(MODEL_PATH)
    # Get input shape
    input_shape = interpreter.get_input_details()[0]['shape']  # e.g., [1,640,640,3]
    # Preprocess
    input_tensor, orig_img = preprocess_image(IMAGE_PATH, input_shape)
    # Inference
    preds = run_inference(interpreter, input_tensor)
    # Post-process
    dets = postprocess(preds, orig_img)
    # Draw
    out_img = draw_detections(orig_img, dets, CLASS_NAMES)
    # Save or show
    cv2.imshow("output.jpg", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
