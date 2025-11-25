class TrainConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "./data/hotdog"
    batch_size = 256
    num_workers = 16
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 0.001
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    num_classes = 2
    pretrained = True