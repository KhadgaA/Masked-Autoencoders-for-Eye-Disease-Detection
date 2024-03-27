import numpy
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.classification import MulticlassCalibrationError

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res



def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
def get_lr(optimizer):
    for parma_group in optimizer.param_groups:
        return parma_group['lr']

def accuracy(outputs,labels):
    _,preds = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))
#
# def top_k(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def training_step( model,batch,loss_func):
    x, y = batch
    out = model(x)
    loss = loss_func(out,y)
    return loss
def training_step_KD(model,teacher_model,batch,loss_func):
    x,_ = batch
    out = model(x)
    # teacher_model.eval()
    teacher_model.eval()
    y = teacher_model(x)
    loss = loss_func(out,y)
    return loss

def validation_step(model,batch,loss_func,ec,**kwargs):
    # batch = batch
    # loss = loss
    # model = model
    x, y = batch
    out = model(x)
    loss = loss_func(out, y)
    acc = accuracy(out,y)
    if ec:
        metric = MulticlassCalibrationError(**kwargs,norm='l1')
        return {'val_loss':loss.detach(),'val_acc':acc, 'ECE': metric(out, y)}
    return {'val_loss':loss.detach(),'val_acc':acc}

def validation_epoch_end(outputs,ec):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    if ec: 
        batch_ec = [x['ECE'] for x in outputs]
        epoch_ec = torch.stack(batch_ec).mean()  
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(),'ECE': epoch_ec.item()}
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end( epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

def evaluate(model,loader,loss_func,ec=False,**kwargs):
    with torch.no_grad():
        model.eval()
        outputs = [validation_step(model,batch,loss_func,ec,**kwargs) for batch in loader]
        return validation_epoch_end(outputs,ec)
def get_output(model,batch):
    model.eval()
    images, _ = batch
    return model(images)

def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # cuda device
    device = torch.device(f'cuda:{opt.cudaid}')

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)

            output = model(input)
            try:
                # compute output
                output,latent= output
            except: 
                pass
            # print(output.shape,target.shape)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = Accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return top1.avg, top5.avg, losses.avg








