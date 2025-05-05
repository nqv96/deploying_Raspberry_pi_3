import torch
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np


interpreter = tflite.Interpreter(model_path="/home/vuong/my_project/human-detection/detection.tflite")
interpreter.allocate_tensors()

# Lấy thông tin về các tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# In thông tin chi tiết về model
print("Input details:", input_details)
print("Output details:", output_details)