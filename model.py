import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_custom_model(num_classes):
    """
    Hàm khởi tạo mô hình kết hợp: 
    - ResNet-18 Pretrained làm Feature Extractor (đóng băng các lớp ẩn)
    - Custom Classifier tự xây dựng ở lớp cuối
    """
    # Tải ResNet-18 pretrained với trọng số mặc định chuẩn
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Đóng băng (freeze) các trọng số phần thân để giữ nguyên khả năng trích xuất đặc trưng
    for param in model.parameters():
        param.requires_grad = False
        
    # Lấy số chiều đầu vào của lớp Fully Connected gốc (thường là 512)
    num_ftrs = model.fc.in_features
    
    # Thay thế lớp Fully Connected gốc bằng Custom Classifier do nhóm tự thiết kế
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),                    # Chống overfitting hiệu quả
        nn.Linear(256, num_classes)         # Đầu ra tương ứng với số lớp của bài toán
    )
    
    return model

if __name__ == "__main__":
    # Test thử khởi tạo mô hình với 3 lớp nhãn
    test_model = get_resnet18_custom_model(num_classes=3)
    print("Khởi tạo mô hình ResNet-18 + Custom CNN thành công!")
    print(test_model)