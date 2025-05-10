import argparse
import torch
import tflite_runtime.interpreter as tflite
import numpy as np
from det_helper import MergeNMS, Yolo3Output
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import psutil
import time

def build_det_helper():
    """Khởi tạo các thành phần xử lý đầu ra của YOLO"""
    nms = MergeNMS.build_from_config({
        "nms_name": "merge",
        "nms_valid_thres": 0.01,
        "nms_thres": 0.45,
        "nms_topk": 400,
        "post_nms": 100,
        "pad_val": -1,
    })

    output_configs = [
        {"num_class": 1, "anchors": [116, 90, 156, 198, 373, 326], "stride": 32, "alloc_size": [128, 128]},
        {"num_class": 1, "anchors": [30, 61, 62, 45, 59, 119], "stride": 16, "alloc_size": None},
        {"num_class": 1, "anchors": [10, 13, 16, 30, 33, 23], "stride": 8, "alloc_size": None},
    ]

    outputs = [Yolo3Output(**cfg).eval() for cfg in output_configs]
    return nms, outputs

def preprocess_image(image):
    image_np = np.array(image)[None, ...]
    image_np = (image_np / 255) * 2 - 1
    return image_np.astype('float32')

def load_and_preprocess_image(image_path, target_size):
    """Tiền xử lý ảnh"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_np = np.array(image)[None, ...]
    image_np = (image_np / 255) * 2 - 1
    return image_np.astype('float32'), image

def detect_and_draw(image_path, interpreter, nms_layer, output_layers):
    """Thực hiện phát hiện và vẽ kết quả"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1:3]

    # Sử dụng hàm preprocess_image có sẵn từ eval_det.py
    image_raw = Image.open(image_path).convert("RGB")
    # ti le anh goc 
    orig_w, orig_h = image_raw.size

    image = Image.open(image_path).convert("RGB")
    image = image.resize(resolution[::-1])  # Resize theo width, height  
    scale_x = orig_w / resolution[1]
    scale_y = orig_h / resolution[0]
    image_np = preprocess_image(image)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()

    # Sử dụng cách xử lý output giống như trong eval_det.py
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                  for i in range(len(output_details))]
    outputs = [torch.from_numpy(d).permute(0, 3, 1, 2).contiguous()
              for d in output_data]
    outputs = [output_layer(output)
              for output_layer, output in zip(output_layers, outputs)]
    # In ra thông tin thô từ YOLO sau khi qua các output layers
    # print("Raw model outputs after Yolo3Output layers:")
    # for i, out in enumerate(outputs):
    #     print(f"Output {i}: shape = {out.shape}")
    #     print(out[0, :5])  # In 5 dòng đầu tiên (batch size 1)
    outputs = torch.cat(outputs, dim=1)

    # NMS và lọc kết quả
    ids, scores, bboxes = nms_layer(outputs)
    threshold = 0.3
    n_positive = (scores > threshold).sum()
    ids = ids[0, :n_positive, 0].numpy()
    bboxes = bboxes[0, :n_positive].numpy()
    scores = scores[0, :n_positive].numpy()
    
    # Chuyển bbox từ resize -> ảnh gốc
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y
    # print("\nResults after NMS and score thresholding:")
    # for i in range(n_positive):
    #     print(f"ID: {ids[i]}, Score: {scores[i]:.2f}, BBox: {bboxes[i]}")
    # Vẽ kết quả
    image_draw = ImageDraw.Draw(image_raw)
    for i,bbox in enumerate(bboxes):
        bbox = [int(x) for x in bbox]
        image_draw.rectangle(bbox, outline="red", width=2)
        image_draw.text((bbox[0], bbox[1] - 10), f"{float(scores[i]):.2f}", fill="red") 
        print("Detected person at:", bbox)
    return image_raw

# def get_model_size(model_path):
#     """Lấy kích thước của model file"""
#     size_bytes = os.path.getsize(model_path)
#     size_mb = size_bytes / (1024 * 1024)
#     return size_mb

# def get_memory_usage():
#     """Lấy thông tin về RAM đang sử dụng"""
#     process = psutil.Process()
#     memory_info = process.memory_info()
#     return memory_info.rss / (1024 * 1024)  # Convert to MB

# def get_cpu_usage():
#     """Lấy thông tin về CPU đang sử dụng"""
#     return psutil.cpu_percent()

# def measure_inference_time(interpreter, input_data):
#     """Đo thời gian inference"""
#     start_time = time.time()
#     interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
#     interpreter.invoke()
#     end_time = time.time()
#     return (end_time - start_time) * 1000  # Convert to milliseconds
#-------------------------Version chay local----------------------------------
# def main():
#     # Load model
#     MODEL_PATH = "/home/vuong/my_project/human-detection/detection.tflite"
    
#     # # Kiểm tra kích thước model
#     # model_size = get_model_size(MODEL_PATH)
#     # print(f"Model size: {model_size:.2f} MB")
    
#     # # Kiểm tra RAM trước khi load model
#     # initial_memory = get_memory_usage()
#     # print(f"Initial memory usage: {initial_memory:.2f} MB")
    
#     interpreter = tflite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()
#     # get input, output resolution
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     input_shape = input_details[0]['shape']
#     output_shape = output_details[0]['shape']
#     resolution = output_shape[1:3]
#     # print(output_shape, output_details)
    
#     # # Kiểm tra RAM sau khi load model
#     # after_load_memory = get_memory_usage()
#     # print(f"Memory usage after loading model: {after_load_memory:.2f} MB")
#     # print(f"Memory increase: {after_load_memory - initial_memory:.2f} MB")
    
#     # # Kiểm tra CPU usage
#     # cpu_usage = get_cpu_usage()
#     # print(f"CPU usage: {cpu_usage}%")

#     # Khởi tạo YOLO helpers
#     nms_layer, output_layers = build_det_helper()

#     # Test images
#     test_images = [
#         "/home/vuong/my_project/human-detection/images_test/art_16.jpg"
#         # "/home/vuong/my_project/human-detection/images_test/image.png",
#         # "/home/vuong/my_project/human-detection/images_test/34919511_122594755296479_501094598928498688_n.jpg"
#     ]
#     # Hiển thị kết quả
#     plt.figure(figsize=(15, 5))
#     for i, image_path in enumerate(test_images, 1):
#         plt.subplot(1, 2, i)
        
#         # Đo thời gian inference
#         image = Image.open(image_path).convert("RGB")
#         input_shape = interpreter.get_input_details()[0]['shape']
#         image = image.resize(input_shape[1:3][::-1])
#         image_np = preprocess_image(image)
#         # inference_time = measure_inference_time(interpreter, image_np)
#         # print(f"Inference time: {inference_time:.2f} ms")
#         result_image = detect_and_draw(image_path, interpreter, nms_layer, output_layers)
#         plt.imshow(result_image)
#         plt.axis('off')
#     plt.show()
#--------------------Version tham so dong lenh-----------------------------

def measure_inference_time(interpreter, input_data):
    start_time = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()
    return (end_time - start_time) * 1000  # ms

def main():
    parser = argparse.ArgumentParser(description="YOLOv3 TinyLite human detection script")
    parser.add_argument("--model", required=True, help = "Path to .tflite model")
    parser.add_argument("--images", nargs="+", help="List of image paths to test")
    # parser.add_argument("--score_thresh", type=float, default=0.3, help="Score threshold for detection")
    parser.add_argument('--show', action='store_true', help="Show result images using matplotlib")
    parser.add_argument('--time', action='store_true', help="Print inference time")
    args = parser.parse_args()
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    nms_layer, output_layers = build_det_helper()
    if args.show:
        plt.figure(figsize=(15, 5))
    for i, image_path in enumerate(args.images, 1):
        image = Image.open(image_path).convert("RGB")
        input_shape = interpreter.get_input_details()[0]['shape']
        image = image.resize(input_shape[1:3][::-1])
        image_np = preprocess_image(image)

        if args.time:
            inference_time = measure_inference_time(interpreter, image_np)
            print(f"Inference time on {os.path.basename(image_path)}: {inference_time:.2f} ms")

        result_image = detect_and_draw(image_path, interpreter, nms_layer, output_layers)

        if args.show:
            plt.subplot(1, len(args.images), i)
            plt.imshow(result_image)
            plt.title(os.path.basename(image_path))
            plt.axis('off')
    if args.show:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
