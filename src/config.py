import torch

class Params:
    def __init__(self):
        self.batch_size = 128
        self.name = "resnet_50"
        self.workers = 16
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        self.num_epochs = 100

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ) 