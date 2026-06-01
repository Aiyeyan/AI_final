import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# 兼容 1通道 和 3通道 的数据集加载
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_rgb=False):
        data = pd.read_csv(csv_file)
        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform
        self.is_rgb = is_rgb

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        from PIL import Image
        image = Image.fromarray(image)
        
        # 如果使用迁移学习网络，需要转为RGB3通道
        if self.is_rgb:
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
        return image, label

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'r-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'g-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线图已保存为 {save_path}")

def get_mobilenet_model(num_classes=25):
    # 使用预训练权重，加速收敛并提升精度
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # 替换最后的分类器层
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 针对 MobileNetV2 的数据增强与预处理 (224x224 RGB)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomRotation(20),   

        transforms.RandomAffine(
            degrees=0,
            translate=(0.1,0.1),
            scale=(0.9,1.1)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter( #均从0.3-》0.5
            brightness=0.5,
            contrast=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = SignLanguageDataset('data/sign_mnist_train.csv', transform=train_transform, is_rgb=True)
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    val_full_dataset = SignLanguageDataset('data/sign_mnist_train.csv', transform=val_transform, is_rgb=True)
    val_dataset = Subset(val_full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 使用多个CPU核心并行加载;锁页内存，加速数据传输到GPU
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = get_mobilenet_model(num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  #lr:0.0005->0.0001 ->0.001
    # 引入余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_acc = 0.0
    epochs = 15

    train_losses, train_accs,val_losses, val_accs = [], [], [], []

    print("Starting Training (MobileNetV2 Transfer Learning)...")
    start_time=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)      # 计算验证损失
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_acc = val_correct / val_total
        epoch_val_loss = val_loss / len(val_loader)
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        val_losses.append(epoch_val_loss)   # 新增
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_train_loss:.4f} "
              f"Train Acc: {epoch_train_acc:.4f} Val Acc: {epoch_val_acc:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_mobilenet_v2.pth')
            print(f"--> Best Model Saved (Accuracy: {epoch_val_acc:.4f})")

    end_time=time.perf_counter()
    total_time=end_time-start_time
    print(f'Total training time: {total_time:.2f} seconds')
    print(f'模型训练完成！总耗时: {total_time // 60:.0f} 分 {total_time % 60:.0f} 秒')
    print(f"Training completed in {end_time-start_time:.2f} seconds")

    plot_training_curves(train_losses, train_accs, val_losses,val_accs)
    
    # --- 端侧部署准备：导出 ONNX 格式 ---
    print("\n[端侧部署] 正在导出 ONNX 模型格式...")
    model.load_state_dict(torch.load('best_mobilenet_v2.pth'))
    model.eval().to('cpu')
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    torch.onnx.export(model, dummy_input, "mobilenet_v2_sign.onnx", 
                      export_params=True, opset_version=11, 
                      input_names=['input'], output_names=['output'])
    print("[端侧部署] ONNX 模型已保存至 mobilenet_v2_sign.onnx")

if __name__ == '__main__':
    train_model()