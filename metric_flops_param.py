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