import torch
from torchvision import transforms
from PIL import Image
from model import get_resnet18_custom_model

def predict_single_image(image_path, model_weight_path="model_best.pth", class_names=None):
    if class_names is None:
        class_names = ['Lớp 1', 'Lớp 2', 'Lớp 3'] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(class_names)
    model = get_resnet18_custom_model(num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    except FileNotFoundError:
        print(f"Không tìm thấy file trọng số '{model_weight_path}'. Hãy chạy file train.py trước!")
        return

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Không tìm thấy file ảnh tại đường dẫn: {image_path}")
        return

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    print("========================================")
    print(f" Ảnh đầu vào: {image_path}")
    print(f" Kết quả dự đoán: {predicted_class}")
    print(f" Độ tin cậy (Confidence): {confidence.item()*100:.2f}%")
    print("========================================")

import sys

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "wifi_signal_image.jpg"
    
    class_names = [
        'wifi_normal',
        'wifi_packet_loss',
        'wifi_bluetooth_interference',
        'wifi_microwave_interference',
        'wifi_co_channel_interference',
        'wifi_congestion',
        'wifi_path_loss_fading',
        'wifi_multipath_distortion'
    ]
    
    predict_single_image(
        image_path=img_path, 
        model_weight_path="model_best.pth", 
        class_names=class_names
    )
