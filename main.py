import argparse

def train(args):
    from trainer.train import main as train_main
    train_main(args)
    
def infer(args):
    from inference.inference import predict
    pred, prob = predict(args.imag_path, args.model_path, device='cpu')
    print(f"Prediction: {pred}, ")
        
def export(args):
    from inference.export import export_onnx, export_torchscript
    export_onnx(args.model_path, args.output_path)
    export_torchscript(args.model_path, args.output_path, device=args.device)
    
def deploy(args):
    import subprocess
    subprocess.run(["python", "-m", "uvicorn", "inference.api:app", "--reload"])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HotDog Recognizer CLI")
    subparser = parser.add_subparsers()
    
    # train subcommand
    parser_train = subparser.add_parser("train", help="Train the model")
    # train argument would be added here
    parser_train.add_argument("--config", type=str, required=True, default= 'config.yaml',help="trainning configuration file path")
    parser_train.set_defaults(func=train)
    
    #inference subcommand
    parser_infer = subparser.add_parser("infer", help="Run inference on an image")
    # inference arguments
    parser_infer.add_argument("--img_path", type=str, required=True, help="Path to the input images")
    parser_infer.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser_infer.add_argument("--device", type=str, default='cpu', help="Device to run inference on (cpu or cuda)")
    parser_infer.set_defaults(func=infer)

    # export subcommand
    parser_export = subparser.add_parser("export", help="Export the model")
    parser_export.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser_export.add_argument("--onnx_path", type=str, required=True, default='export/hotdog.onnx', help="Path to save the exported model")
    parser_export.add_argument("--torchscript_path", type=str, required=True, default='export/hotdog.pt', help="path to save the exported model")
    parser_export.add_argument("--device", type=str, default='cuda', help="Device to run export on (cpu or cuda)")    
    parser_export.set_defaults(func=export)
    
    # deploy subcommand
    parser_deploy = subparser.add_parser("deploy", help="Deploy the model as as API Service")
    parser_deploy.set_defaults(func=deploy)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
            