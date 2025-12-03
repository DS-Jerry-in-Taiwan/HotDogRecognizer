import torch
from model.models import HotDogsRecognizeModel

def export_onnx(model_path, export_path, device='cuda' ,num_classes=2):
    # Load the trained model
    model = HotDogsRecognizeModel.load(model_path, device=device, num_classes=num_classes)
    model.eval()
    
    # create a dummy input tensor with the appropriate size
    dummy_input = torch.randn(1, 3,224, 224).to(device)
    
    # Export the model to ONNX format
    torch.onnx.export(
        model, dummy_input, export_path,
        input_names=['input'], output_names=['output'],
        opset_version=12, do_constant_folding=True
    )
    print(f"Model exported to {export_path}")
    
    
def export_torchscript(model_path, export_path, device='cuda' ,num_classes=2):
    # Load the trained model
    model = HotDogsRecognizeModel.load(model_path, device=device, num_classes=num_classes)
    model.eval()
    # export the model to TorchScript format
    scripted_model = torch.jit.script(model)
    scripted_model.save(export_path)
    print(f"TorchScript model exported to {export_path}")
    
if __name__ == "__main__":
    export_onnx("../checkpoints/hotdog_model.pth", "../exported_models/hotdog_model.onnx", device='cuda', num_classes=2)
    export_torchscript("../checkpoints/hotdog_model.pth", "../exported_models/hotdog_model.pt", device='cuda', num_classes=2)