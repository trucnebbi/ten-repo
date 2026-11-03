import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_custom_model(num_classes):
    """
    Hàm khởi tạo mô hình kết hợp: 
    - ResNet-18 Pretrained làm Feature Extractor (đóng băng các lớp ẩn)
    - Custom Classifier tự xây dựng ở lớp cuối
    """
    
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),                   
        nn.Linear(256, num_classes)         
    )
    
    return model

if __name__ == "__main__":
    test_model = get_resnet18_custom_model(num_classes=3)
    print("Khởi tạo mô hình ResNet-18 + Custom CNN thành công!")
    print(test_model)
