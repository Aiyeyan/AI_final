import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# 引入端侧部署专用轻量化推理引擎（需要提前 pip install onnxruntime）
import onnxruntime as ort

app = Flask(__name__)

# 定义手语字母映射 (A-Y, 缺少J)
CLASS_NAMES = [chr(65 + i) for i in range(25)]

# 1. 载入端侧部署的中间件模型 (ONNX模型)
ONNX_PATH = "mobilenet_v2_sign.onnx"
if not os.path.exists(ONNX_PATH):
    print(f"警告: 未找到 {ONNX_PATH}，请先运行 model_train.py 导出该模型！")
else:
    # 创建推理会话，完全运行在 CPU 上，模拟端侧无GPU环境
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

def preprocess_image(image_base64):
    """前端传来的Base64图片预处理，适配MobileNetV2输入"""
    # 解码base64图片
    img_data = base64.b64decode(image_base64.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 读入 BGR 图像
    
    # 镜像翻转、裁剪及缩放
    img = cv2.flip(img, 1)
    # 模拟真实部署时的中央区域裁剪 (Center Crop)
    h, w, _ = img.shape
    crop_size = min(h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    roi = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # 缩放到 MobileNet 标准输入 224x224 并转为 RGB
    roi_resized = cv2.resize(roi, (224, 224))
    rgb_img = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # 归一化处理 (完美匹配 PyTorch 的 transforms.Normalize)
    img_data = rgb_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # 调整通道顺序 [H, W, C] -> [C, H, W]，并增加 Batch 维度 -> [1, C, H, W]
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

@app.route('/')
def index():
    """系统前端主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """后台推理核心API"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': '未接收到图像数据'}), 400
            
        # 1. 前端图像预处理
        input_tensor = preprocess_image(data['image'])
        
        # 2. 使用 ONNX 引擎执行纯端侧 CPU 推理
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        import time
        start_time = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000 # 毫秒单位
        
        # 3. 解析模型输出结果
        logits = ort_outs[0][0]
        # Softmax 转化为概率分布
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_char = CLASS_NAMES[pred_idx]
        
        return jsonify({
            'status': 'success',
            'prediction': pred_char,
            'confidence': f"{confidence:.1%}",
            'latency': f"{inference_time:.1f}ms"
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(i)}), 500

if __name__ == '__main__':
    # 启动本地服务，控制台打印提示
    print("*" * 60)
    print("手语识别系统后台已启动！请在浏览器中访问: http://127.0.0.1:5000")
    print("*" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)