import torch
from torchvision import transforms
from PIL import Image
from model import get_resnet18_custom_model

def predict_single_image(image_path, model_weight_path="model_best.pth", class_names=None):
    if class_names is None:
        class_names = ['Lớp 1', 'Lớp 2', 'Lớp 3'] # Thay đổi tên nhãn tương ứng với dữ liệu của bạn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo kiến trúc mô hình
    num_classes = len(class_names)
    model = get_resnet18_custom_model(num_classes=num_classes)
    
    # Tải trọng số đã huấn luyện
    try:
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    except FileNotFoundError:
        print(f"Không tìm thấy file trọng số '{model_weight_path}'. Hãy chạy file train.py trước!")
        return

    model = model.to(device)
    model.eval()

    # 2. Xử lý ảnh đầu vào
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

    # 3. Dự đoán
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

if __name__ == "__main__":
    # Ví dụ gọi hàm dự đoán một ảnh test
    # predict_single_image("test_image.jpg", class_names=['Normal', 'Defect_1', 'Defect_2'])
    print("Hãy điền đường dẫn ảnh cụ thể vào hàm predict_single_image để chạy thử.")