import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import time
from model_train import get_mobilenet_model

DEVICE = torch.device("cpu") # 强制使用CPU模拟端侧推理
#CLASS_NAMES = [chr(65 + i) for i in range(25)]
# 同样修正标签映射
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I',
    'SKIP_J', 'K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y'
]

ROI_TOP, ROI_LEFT = 100, 200
ROI_SIZE = 224  # 直接使用224匹配MobileNet输入

def load_quantized_model():
    # 初始化浮点模型框架
    model = get_mobilenet_model(num_classes=25)
    # 直接量化框架
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    # 加载量化后的权重
    quantized_model.load_state_dict(torch.load('quantized_mobilenet.pth', map_location=DEVICE))
    quantized_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return quantized_model, transform

def main():
    model, transform = load_quantized_model()
    print("已加载端侧量化模型: INT8 MobileNetV2 (按 'q' 退出)")

    cap = cv2.VideoCapture(0)
    
    # 用于计算 FPS
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        #h, w = frame.shape[:2]

        x1, y1 = ROI_LEFT, ROI_TOP
        x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        roi = frame[y1:y2, x1:x2]
        
        # 预处理
        #rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        #pil_img = Image.fromarray(rgb)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


        # 转回 PIL
        #pil_img = Image.fromarray(thresh).convert("RGB")
        # 直接转回三通道 RGB 并走 transform
        pil_img = Image.fromarray(gray).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0)

        # 推理并计算时间
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1).squeeze()

        top1_prob, top1_idx = torch.topk(probs, 1)
        pred_char = CLASS_NAMES[top1_idx.item()]
        
        # 计算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 显示文本
        txt = f"Pred: {pred_char} ({top1_prob.item():.1%})"
        fps_txt = f"FPS: {fps:.1f} (INT8 Quant)"
        
        cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, fps_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

        cv2.imshow("Edge-Device Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()