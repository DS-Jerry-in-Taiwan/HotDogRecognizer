import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_metrics(train_loss, train_acss, test_loss, test_acss):
    """
    visualize training and testing metrics
    """
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acss, label='Train Accuracy')
    plt.plot(epochs, test_acss, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    

def show_wrong_predictions(images, y_true, y_pred, class_names, max_show=8):
    wrong_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
    show_num = min(len(wrong_idx), max_show)
    if show_num == 0:
        print("No wrong predictions to show.")
        return
    plt.figure(figsize=(12, 3))
    for i in range(show_num):
        idx = wrong_idx[i]
        plt.subplot(1, show_num, i+1)
        img = images[idx]
        if hasattr(img, 'permute'):  # torch tensor
            img = img.permute(1,2,0).cpu().numpy()
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img = img.clip(0, 1)
        plt.imshow(img)
        plt.title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()