import torch
import numpy as np


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                # count correct number
                acc_sum += (net(X.to(device)).argmax(dim=1) ==
                            y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1)
                                == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def evaluate_class_accuracy(data_iter, net, num_class=10, device=None):
    acc_sum, n = np.zeros(num_class, dtype=np.float32), np.zeros(num_class, dtype=np.float32)
    with torch.no_grad():
        net.eval()
        for X, y in data_iter:
            y_hat = net(X.to(device)).argmax(dim=1)
            acc_sum += ((y+1)*(y==y_hat)).bincount(minlength=num_class+1)[1:].numpy()
            n += y.bincount(minlength=num_class).numpy()
    net.train()       
    return np.divide(acc_sum, n)
