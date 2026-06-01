import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# 引入端侧部署专用轻量化推理引擎（需要提前 pip install onnxruntime）
import onnxruntime as ort

app = Flask(__name__)

# 定义手语字母映射 (A-Y, 缺少J)
#CLASS_NAMES = [chr(65 + i) for i in range(25)]
# 1. 严格对齐 25 个输出槽位，填充被跳过的 'J'（可以随便写个占位符，防止索引错位）
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I',
    'SKIP_J', # 对应标签 9（Sign Language MNIST 中缺失）
    'K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y'
]

# 1. 载入端侧部署的中间件模型 (ONNX模型)
ONNX_PATH = "mobilenet_v2_sign.onnx"
if not os.path.exists(ONNX_PATH):
    print(f"警告: 未找到 {ONNX_PATH}，请先运行 model_train.py 导出该模型！")
else:
    # 创建推理会话，完全运行在 CPU 上，模拟端侧无GPU环境
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

def preprocess_image(image_base64):
    """
    像素级对齐版：完美还原 model_train.py 训练集的数据分布
    剔除导致特征严重畸变的 CLAHE 和高斯模糊，只做纯净的单通道克隆
    """
    # 解码前端 Base64 图像
    img_data = base64.b64decode(image_base64.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 读入为 BGR
    #print(f"原始图像尺寸: {img.shape}，像素范围：{img.min()}~ {img.max()}") # 输出原始图像尺寸，便于调试
    
    roi=img
    
    #  核心对齐：先转灰度图（抹除肤色、环境光干扰，回归 MNIST 灰度本质）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 将单通道灰度图复制 3 份堆叠成三通道 RGB
    rgb_simulated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 5. 缩放到 224x224 (MobileNetV2 标准输入尺寸)
    roi_resized = cv2.resize(rgb_simulated, (224, 224))
    
    # 6. 归一化并进行标准的 ImageNet 均值/标准差归一化 (必须与 model_train.py 严格一致)
    img_data = roi_resized.astype(np.float32) / 255.0
    #print(f"归一化后均值: {img_data.mean():.3f}, 标准差: {img_data.std():.3f}")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # 7. 调整通道维度 [H, W, C] -> [C, H, W] 并增加 Batch 维度 -> [1, C, H, W]
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
        
        # 在解析出 pred_char 和 confidence 后，做一层过滤：
        if confidence < 0.40:  # 门槛设为 40%
            return jsonify({
                'status': 'success',
                'prediction': '检测中...', # 置信度太低时不瞎猜
                'confidence': f"{confidence:.1%}",
                'latency': f"{inference_time:.1f}ms"
            })
        else:
            return jsonify({
                'status': 'success',
                'prediction': pred_char,
                'confidence': f"{confidence:.1%}",
                'latency': f"{inference_time:.1f}ms"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # 启动本地服务，控制台打印提示
    print("*" * 60)
    print("手语识别系统后台已启动！请在浏览器中访问: http://127.0.0.1:5000")
    print("*" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)