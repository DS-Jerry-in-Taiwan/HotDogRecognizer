from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    """
    wrapper class for TensorBoard SummaryWriter
    """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log_metrics(self, metrics: dict, epoch: int, prefix: str = ""):
        """
        record multiple metrics to TensorBoard
        """
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, epoch)
            
    def close(self):
        """
        close the SummaryWriter
        """
        self.writer.close()