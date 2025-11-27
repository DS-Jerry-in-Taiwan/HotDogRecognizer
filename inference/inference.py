import torch
from PIL import Image
from torchvision import transforms
from model.models import HotDogsRecognizeModel

def preprocess_image(image):
    """
    Preprocess the input image for model prediction
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]
    )
    img = Image.open(image).convert('RGB')
    return transform(img).unsqueeze(0)

def predict(img_path, model_path, device='cpu'):
    """
    Predict if the image is a hotdog or not
    """
    model=HotDogsRecognizeModel.load(model_path, device=device)
    model.eval()
    img_tensor = preprocess_image(img_path).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).cpu().numpy()
    return pred, prob.squeeze().cpu().numpy()

def batch_predict(img_paths, model_path, device='cpu'):
    model =HotDogsRecognizeModel.load(model_path, device=device)
    model.eval()
    results = []
    for img_path in img_paths:
        img_tensor = preprocess_image(img_path).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
        results.append((img_path, pred, prob.squeeze().cpu().numpy()))
    return results

if __name__ == "__main__":
    # 單張圖片推論
    pred, prob = predict("test.jpg", "checkpoints/model_epoch_100.pth", device='cpu')
    print("預測類別:", pred, "機率分布:", prob)