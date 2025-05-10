'''
main.py use for deploying detection.tflite on Raspberry Pi 3 
'''
import torch
import tflite_runtime.interpreter as tflite
import numpy as np
from det_helper import MergeNMS, Yolo3Output
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import psutil
import time
import argparse
from datetime import datetime

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
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resolution[::-1])  # Resize theo width, height
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
    outputs = torch.cat(outputs, dim=1)

    # NMS và lọc kết quả
    ids, scores, bboxes = nms_layer(outputs)
    threshold = 0.3
    n_positive = (scores > threshold).sum()
    ids = ids[0, :n_positive, 0].numpy()
    bboxes = bboxes[0, :n_positive].numpy()

    # Vẽ kết quả
    image_draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        bbox = [int(x) for x in bbox]
        image_draw.rectangle(bbox, outline="red", width=2)
        # print("Detected person at:", bbox)
    return image

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

def measure_inference_time(interpreter, input_data):
    """Đo thời gian inference"""
    start_time = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds

def save_detection_result(image, bboxes, output_dir):
    """Lưu kết quả phát hiện"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
    image.save(output_path)
    return output_path

def run_detection(interpreter, nms_layer, output_layers, image_path, output_dir, show_result=False):
    """Chạy phát hiện trên một ảnh"""
    try:
        # Đo thời gian inference
        image = Image.open(image_path).convert("RGB")
        input_shape = interpreter.get_input_details()[0]['shape']
        image = image.resize(input_shape[1:3][::-1])
        image_np = preprocess_image(image)
        
        # Thực hiện inference
        inference_time = measure_inference_time(interpreter, image_np)
        print(f"Inference time: {inference_time:.2f} ms")
        
        # Xử lý kết quả
        result_image = detect_and_draw(image_path, interpreter, nms_layer, output_layers)
        
        # Lưu kết quả
        output_path = save_detection_result(result_image, None, output_dir)
        print(f"Saved result to: {output_path}")
        
        if show_result:
            plt.figure(figsize=(10, 10))
            plt.imshow(result_image)
            plt.axis('off')
            plt.show()
            
        return True
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Human Detection on Raspberry Pi')
    parser.add_argument('--model', type=str, default='detection.tflite',
                      help='Path to TFLite model')
    parser.add_argument('--input', type=str, default='images_test',
                      help='Input image or directory')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--show', action='store_true',
                      help='Show detection results')
    parser.add_argument('--continuous', action='store_true',
                      help='Run in continuous mode')
    args = parser.parse_args()

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(args.output, exist_ok=True)

    # # Load model
    # print(f"Loading model: {args.model}")
    # model_size = get_model_size(args.model)
    # print(f"Model size: {model_size:.2f} MB")
    
    # initial_memory = get_memory_usage()
    # print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    
    # after_load_memory = get_memory_usage()
    # print(f"Memory usage after loading model: {after_load_memory:.2f} MB")
    # print(f"Memory increase: {after_load_memory - initial_memory:.2f} MB")
    
    # cpu_usage = get_cpu_usage()
    # print(f"CPU usage: {cpu_usage}%")

    # Khởi tạo YOLO helpers
    nms_layer, output_layers = build_det_helper()

    if args.continuous:
        print("Running in continuous mode...")
        while True:
            try:
                # Tìm tất cả ảnh trong thư mục input
                image_files = [f for f in os.listdir(args.input) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for image_file in image_files:
                    image_path = os.path.join(args.input, image_file)
                    print(f"\nProcessing: {image_path}")
                    run_detection(interpreter, nms_layer, output_layers, 
                                image_path, args.output, args.show)
                
                time.sleep(1)  # Đợi 1 giây trước khi quét lại
                
            except KeyboardInterrupt:
                print("\nStopping continuous mode...")
                break
            except Exception as e:
                print(f"Error in continuous mode: {str(e)}")
                time.sleep(1)
    else:
        # Xử lý một ảnh hoặc thư mục
        if os.path.isdir(args.input):
            image_files = [f for f in os.listdir(args.input) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                image_path = os.path.join(args.input, image_file)
                print(f"\nProcessing: {image_path}")
                run_detection(interpreter, nms_layer, output_layers, 
                            image_path, args.output, args.show)
        else:
            print(f"\nProcessing: {args.input}")
            run_detection(interpreter, nms_layer, output_layers, 
                        args.input, args.output, args.show)

if __name__ == "__main__":
    main()
