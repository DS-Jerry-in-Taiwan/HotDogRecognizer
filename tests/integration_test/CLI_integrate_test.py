
import subprocess

def test_cli_train_help():
    result = subprocess.run(["python", "main.py", "train", "--help"], capture_output=True, text=True)
    assert "usage" in result.stdout

def test_cli_infer_help():
    result = subprocess.run(["python", "main.py", "infer", "--help"], capture_output=True, text=True)
    assert "usage" in result.stdout