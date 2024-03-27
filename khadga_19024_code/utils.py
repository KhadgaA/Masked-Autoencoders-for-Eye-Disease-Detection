import torch
import numpy as np
import time

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def logger(name, history, other_info=None,all= False):
    epochs = len(history) - 1
    train_loss = history[-1]['train_loss']
    val_loss = history[-1]['val_loss']
    val_acc = history[-1]['val_acc']
    accuracies = [x['val_acc'] for x in history]
    max_acc_epoch = np.argmax(accuracies)
    max_acc_loss = history[max_acc_epoch]['val_loss']
    max_acc = history[max_acc_epoch]['val_acc']

    with open(name + '.txt', 'a') as log:
        log.write(
            f' {time.time()} :: epochs: {epochs},train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}, max_acc_epoch: {max_acc_epoch}, max_acc_loss: {max_acc_loss}, max_acc: {max_acc}, other_info: {other_info}\n')
        if all:
         log.write(str(history))
