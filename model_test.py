import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms, models
import os

from model_train import SignLanguageDataset, get_mobilenet_model

def test():
    # 测试环境优先使用 CPU，模拟真实的端侧/移动端设备环境
    device = torch.device("cpu") 
    print(f"Testing on device: {device} (Simulating Edge Environment)")
    test_csv = 'data/sign_mnist_test.csv'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SignLanguageDataset(test_csv, transform=transform, is_rgb=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载浮点模型
    model = get_mobilenet_model(num_classes=25)
    model.load_state_dict(torch.load('best_mobilenet_v2.pth', map_location=device))
    model.eval()

    # --- 核心优化：动态量化（Dynamic Quantization）端侧优化 ---
    print("\n[端侧优化] 正在执行 INT8 动态量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    model_size_fp32 = os.path.getsize('best_mobilenet_v2.pth') / (1024 * 1024)
    torch.save(quantized_model.state_dict(), 'quantized_mobilenet.pth')
    model_size_int8 = os.path.getsize('quantized_mobilenet.pth') / (1024 * 1024)
    print(f"[端侧优化] 模型体积压缩: {model_size_fp32:.2f} MB -> {model_size_int8:.2f} MB")

    y_true, y_pred, error_images = [], [], []
    total_time = 0.0
    num_samples = len(loader)

    print("\n开始测试量化模型...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            start_time = time.time()
            outputs = quantized_model(images)
            end_time = time.time()
            
            # 跳过前几次的热身迭代以计算稳定的推理时间
            if i > 10:
                total_time += (end_time - start_time)
                
            pred = outputs.argmax(1).item()
            true = labels.item()
            
            y_true.append(true)
            y_pred.append(pred)
            
            if pred != true and len(error_images) < 5:
                error_images.append((images[0], true, pred))

    avg_inference_time = total_time / (num_samples - 11)
    print(f"\n量化模型平均单张推理耗时: {avg_inference_time:.4f} 秒 ({1/avg_inference_time:.1f} FPS) (基于CPU)")
    
    print("\nMobileNetV2 (Quantized) - 分类报告:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Quantized MobileNetV2')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    test()