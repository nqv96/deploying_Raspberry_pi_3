from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
# Load a COCO-pretrained YOLOv5n model
# model = YOLO("yolov5n.pt")

# MODEL_PATH = "/home/vuong/my_project/human-detection/yolov5n-fp16.tflite"
# image_path = "/home/vuong/my_project/human-detection/images_test/art_16.jpg"
# # Display model information (optional)
# # model.info()
# # img = Image.open(image_path)
# # results = model("/home/vuong/my_project/human-detection/images_test/art_16.jpg")
# # results.show()
# interpreter = tflite.Interpreter(model_path=MODEL_PATH)
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# resolution = input_details[0]['shape'][1:3]
# print(input_details, resolution)
# print(output_details)

num_classes = 1
conf_threshold = 0.1
nms_threshold = 0.45
input_size = 640  # Kích thước ảnh đầu vào của mô hình


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / np.clip(union_area, a_min=1e-6, a_max=None)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def custom_nms(boxes, scores, iou_threshold=0.45, merge=False):
    """
    boxes: np.ndarray (N,4) [x1,y1,x2,y2]
    scores: np.ndarray (N,)
    return: list of kept indices
    """
    if boxes.shape[0] == 0:
        return []

    # sort by score descending
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]
        ious = compute_iou(boxes[i], boxes[rest])

        if merge:
            # merge weighted average vào box i
            to_merge = rest[ious > iou_threshold]
            if to_merge.size > 0:
                weights = scores[to_merge][:, None]
                merged_box = (boxes[to_merge] * weights).sum(0) / weights.sum()
                # cập nhật box i
                boxes[i] = merged_box

        # chỉ giữ những box có IoU <= threshold
        keep_mask = ious < iou_threshold
        order = rest[keep_mask]

    return keep

def process_yolo_output(pred, conf_thresh=0.25, iou_thresh=0.45):
    """
    pred: np.ndarray (25200, 85)
    Chỉ detect class 'person' (class_id = 0)
    """
    # 1) tách box + objectness + class scores
    boxes = pred[:, :4]            # [cx, cy, w, h], shape (25200,4)
    objectness = pred[:, 4]        # shape (25200,)
    class_scores = pred[:, 5:]     # shape (25200, 80)

    # 2) chỉ lấy score của class 0 ('person')
    person_score = class_scores[:, 0]  # shape (25200,)

    # 3) compute confidence = objectness * class_score
    scores = sigmoid(objectness) * sigmoid(person_score)  # shape (25200,)

    # 4) lọc theo conf_thresh
    mask = scores > conf_thresh
    boxes = boxes[mask]     # shape (M,4)
    scores = scores[mask]   # shape (M,)

    if boxes.shape[0] == 0:
        return np.zeros((0,4)), np.zeros((0,))

    # 5) chuyển [cx,cy,w,h] -> [x1,y1,x2,y2]
    cx, cy, w, h = boxes.T
    converted = np.stack([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ], axis=1)  # shape (M,4)

    # 6) NMS (merge=True nếu muốn merge)
    keep_idxs = custom_nms(converted, scores, iou_threshold=iou_thresh, merge=False)

    # 7) trả về boxes và scores đã lọc
    return converted[keep_idxs], scores[keep_idxs]


def preprocess_image(image):
    """
    image: np.ndarray HxWx3 (uint8)
    trả về np.ndarray (1, H, W, 3) float32
    """
    image_np = image.astype('float32') / 255.0
    return np.expand_dims(image_np, axis=0)


def process_frame(frame, interpreter, input_details, output_details, original_size):
    """
    frame: np.ndarray BGR
    original_size: (height, width)
    """
    input_size = input_details[0]['shape'][1]  # e.g. 640
    # 1) To RGB + resize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
    # 2) Preprocess
    frame_np = preprocess_image(frame_resized)
    # 3) Inference
    interpreter.set_tensor(input_details[0]['index'], frame_np)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])  # (1,25200,85)
    preds = raw_output[0]  # (25200,85)
    # 4) Post-process
    final_boxes, final_scores = process_yolo_output(preds)
    # 5) scale về kích thước gốc
    orig_h, orig_w = original_size
    scale_w = orig_w / input_size
    scale_h = orig_h / input_size
    final_boxes[:, [0, 2]] *= scale_w
    final_boxes[:, [1, 3]] *= scale_h
    return final_boxes.astype(int), final_scores


def draw_boxes(image_raw, bboxes, scores):
    if len(bboxes) == 0:
        print("No person detected.")
        return image_raw
    image_draw = ImageDraw.Draw(image_raw)
    for i, bbox in enumerate(bboxes):
        bbox = [int(x) for x in bbox]
        image_draw.rectangle(bbox, outline="red", width=2)
        text_y = max(bbox[1] - 10, 0)
        image_draw.text((bbox[0], text_y), f"{float(scores[i]):.2f}", fill="red")
        print("Detected person at:", bbox)
    return image_raw

def main():
    parser =argparse.ArgumentParser(description="Real-time human detection using yolo5n.tflite model")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--images", nargs="+", help="List of image paths to test")
    parser.add_argument('--show', action='store_true', help="Show result images using matplotlib")
    args = parser.parse_args()
    
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)
    for img_path in args.images:
        # Load ảnh
        img_raw = Image.open(img_path).convert("RGB")
        img_raw = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        frame = np.array(img_raw)
        original_size = frame.shape[:2]  # (height, width)

        # Chạy model
        boxes, scores = process_frame(frame, interpreter, input_details, output_details, original_size)

        # Vẽ kết quả
        result_img = draw_boxes(img_raw, boxes, scores)

        # Hiển thị nếu cần
        if args.show:
        # Chuyển PIL Image về BGR array
            vis = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
            cv2.imshow("Detection", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
if __name__ == "__main__":
    main()