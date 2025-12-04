import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import random
import matplotlib.pyplot as plt
from trainer.data import get_data_loaders

def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = img.clip(0, 1)
    return img


def on_click(event):
    if event.button == 1:  # 左鍵換下一張
        index[0] = (index[0] + 1) % len(sample_imgs)
        show_image(index[0])
    elif event.button == 3:  # 右鍵關閉
        plt.close(fig)

def show_image(i):
    ax.clear()
    ax.imshow(sample_imgs[i])
    ax.set_title(f'Image {sample_labels[i]}')
    ax.axis('off')
    fig.canvas.draw()

# defin data directory and batch size
data_dir = "./data/hotdog"
batch_size = 128

# get data loaders
train_loader, test_loader = get_data_loaders(data_dir, batch_size, num_workers=4)

# stochastically sample a batch from train_loader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# stochastically pick up to 8 images from the batch
N =8
indices = random.sample(range(len(images)), N)
sample_imgs = [unnormalize(images[i]) for i in indices]
sample_labels = [labels[i].item() for i in indices]

# plot the sampled images
fig, ax = plt.subplots()
index = [0]  # 用 list 包裝以便在事件中修改

show_image(index[0])
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

