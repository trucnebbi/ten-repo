import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_resnet18_custom_model
import os

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Đang sử dụng thiết bị: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),      
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = './dataset/train'
    if not os.path.exists(train_dir):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {train_dir}!")
        print("Vui lòng tạo thư mục dataset/train và chia các lớp thành các thư mục con.")
        return

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    num_classes = len(train_dataset.classes)
    print(f"-> Số lớp phân loại: {num_classes} ({train_dataset.classes})")

    # 4. Khởi tạo mô hình
    model = get_resnet18_custom_model(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 5  
    print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ===")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

    torch.save(model.state_dict(), "model_best.pth")
    print("=== HUẤN LUYỆN HOÀN TẤT! Đã lưu trọng số vào 'model_best.pth' ===")

if __name__ == "__main__":
    train_model()
