from trainer.config import TrainConfig

def test_config_values():
    cfg = TrainConfig
    print("data_dir:", cfg.data_dir)
    print("batch_size:", cfg.batch_size)
    print("num_workers:", cfg.num_workers)
    print("num_epochs:", cfg.num_epochs)
    print("learning_rate:", cfg.learning_rate)
    print("weight_decay:", cfg.weight_decay)
    print("log_dir:", cfg.log_dir)
    print("checkpoint_dir:", cfg.checkpoint_dir)
    print("num_classes:", cfg.num_classes)
    print("pretrained:", cfg.pretrained)

if __name__ == "__main__":
    test_config_values()