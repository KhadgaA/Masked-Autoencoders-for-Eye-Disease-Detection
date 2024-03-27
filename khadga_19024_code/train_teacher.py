

import imp
import math
import os
import sys
import time
from turtle import forward
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import helper as H
import utils as U
from model_new import *

import torchvision.models as Models
from torchvision import transforms
from torchvision import datasets

from mae_imagenet.build_imnet_mae import *
from data.ODIR import get_odir_dataloaders
# from data.ODIR_32p import get_odir_dataloaders
from loss_funcs import *
from helper import Accuracy, AverageMeter


def augment(generator, images, opt):
    loss, predicted_img, mask, latents = generator(images, mask_ratio=opt.maskratio)
    predicted_img = generator.unpatchify(predicted_img)
    
    mask = mask.unsqueeze(-1).repeat(1, 1, generator.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = generator.unpatchify(mask)  # 1 is removing, 0 is keeping
    
    augment_image = predicted_img * mask + images * (1 - mask)
    # augment_image = 0.8 * augment_image + 0.2 * images
    return augment_image, latents

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--cudaid', type=int, default=0, help='cuda device id')
    parser.add_argument('-maxstp',action = 'store_true', default=False, 
                        help='choose if generator performs maximation step or not')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')


    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD',choices=['SGD', 'Adam'] ,help='optimizer to use, SGD or ADAM')
    parser.add_argument('--scheduler', type=str, default='multistep',choices=['multistep','cosineAnnealing'] ,help='choose a lr schedular')

    # dataset
    parser.add_argument('--dataset', type=str, default='ODIR', choices=['ODIR'], help='dataset')
    
    parser.add_argument('--gen_epochs', type=int, default=10, help='how many epochs to train the generator for, each time')#20
    parser.add_argument('--train_epochs', type=int, default=25, help='how many training epochs for student before the generator training')#50
  
    parser.add_argument('--model_t', type=str, default='resnet50', choices=['resnet50','resnet50_mod','resnet18','wrn_50_2','vgg16','densenet121','densenet201','shuv2_x1_0','shuv2_x2_0'], help='teacher model')

    parser.add_argument('--weights', type=str, default='imagenet', choices=['none','imagenet','odir','odir_32p'], help='teacher model')


 
    
    # Auto Version folder
    parser.add_argument('--run_name', type=str, default= 'finetune50e', help= 'name of the run (default "" )')
    parser.add_argument('--trial', type=str, default= '0', help= 'trial id of the run (default 0 : automatically creates new folder for each run,)')


    opt = parser.parse_args()

    
    if opt.trial != '0':
        opt.model_path = f'./save/teacher_model/{opt.dataset},{opt.model_t},{opt.run_name},v{str(opt.trial)}'
        opt.tb_path = f"./save/teacher_tensorboards/{opt.dataset},{opt.model_t},{opt.run_name},v{str(opt.trial)}"
    else: 
        exists = True
        while exists:
            opt.trial = str(int(opt.trial) + 1)
            opt.model_path = f'./save/teacher_model/{opt.dataset},{opt.model_t},{opt.run_name},v{str(opt.trial)}'
            opt.tb_path = f"./save/teacher_tensorboards/{opt.dataset},{opt.model_t},{opt.run_name},v{str(opt.trial)}"
            exists = os.path.exists(opt.model_path)
             

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f'{opt.model_t}_{opt.trial}'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    #
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
        # os.makedirs(os.path.join(opt.save_folder, 'generator'))
        os.makedirs(os.path.join(opt.save_folder, 'student'))

    return opt


def main(**kwargs):
    opt = parse_option()
    # cuda device
    device = torch.device(f'cuda:{opt.cudaid}')
    # tensorboard logger
    writer = SummaryWriter(opt.tb_folder, flush_secs=2)
    dataset = opt.dataset
    
    if dataset == 'ODIR':
        print('loading ODIR')      
        train_loader, val_loader,n_data = get_odir_dataloaders(batch_size=opt.batch_size,val_batch_size=opt.batch_size//2,
                                                            num_workers=opt.num_workers)
        num_classes = 8
        print('loaded ODIR')
        gen = './mae_imagenet/output_dir_mae_finetune/checkpoint-last.pth'
        
        if opt.model_t == 'vitb_clf':
            pass
        else:
            if opt.weights == 'none':
                model_s = model_dict[opt.model_t](num_classes=num_classes)
            elif opt.weights == 'imagenet':
                model_t_weights = model_path_dict[opt.model_t + f'_{opt.weights}']
                model_s = model_dict[opt.model_t](num_classes=num_classes,weights = model_t_weights)
            # elif opt.weights == 'odir':
            #     model_t_weights = model_path_dict[opt.model_t + f'_{opt.weights}']
            #     model_s = model_dict[opt.model_t](num_classes=num_classes)
            #     model_s.load_state_dict(torch.load(model_t_weights)['model'])
            else:
                print('Weights not found/available')
                sys.exit()
            print(model_s)            
            print(numel(model_s,True))
            

            
    else:
        print('dataset not found')
        sys.exit()
        
    criterion_cls = nn.CrossEntropyLoss().to(device)
    
     # optimizer
    if opt.optimizer =='SGD':
        optimizer = optim.SGD(model_s.parameters(), lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,nesterov=True)
    elif opt.optimizer =='Adam':
        optimizer = optim.Adam(model_s.parameters(), lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    if opt.scheduler =='cosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = opt.epochs)
    if opt.maxstp:
        pass

    # Gpu check 
    if torch.cuda.is_available():
        # module_list.to(device)
        # criterion_list.to(device)
        model_s.to(device)
        cudnn.benchmark = True
        
    val_loader = U.DeviceDataLoader(val_loader, device)
    # result = H.evaluate(model_s, val_loader, criterion_cls)
    # print('teacher accuracy', result['val_acc'])
    grt = 0 # local generator training epoch
    tot = 0 # global generator training epoch
    e = 1 # starting epoch, handy when resuming training
    ## resume/load chkpt code.. 
    
    best_acc = 0 
    save_epochs = [50,100, 150, 200]
    print('traning start')
    for epoch in range(e, opt.epochs + 1):

        model_s.train() 
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        if opt.scheduler =='multistep':
            H.adjust_learning_rate(epoch, opt, optimizer)
        writer.add_scalar('lr_student', H.get_lr(optimizer), epoch)
        # print(grt,opt.train_epochs)
        if grt == opt.train_epochs:
            grt = 0
            for ep in range(1,opt.gen_epochs+1):
                    if opt.maxstp == 'True':
                        # Generator maximization step
                        # Nothing for now
                        pass
            tot +=1
  
        for data in tqdm(train_loader,desc = f'epoch: {epoch}/{opt.epochs} Stu'):
            # continue
            # ===================backward=====================
            optimizer.zero_grad()
            
            input, target, index = data
            input = input.float()
            input = input.to(device)
            target = target.to(device)
            index = index.to(device)
           
                
                
            with torch.cuda.amp.autocast():
                
                logit_t = model_s(input)
                try:
                    logit_t,latents = logit_t
                except:
                    pass
                # print(logit_t.shape)           
                loss_cls = criterion_cls(logit_t,target)
                
                loss = loss_cls
            # print('loss',loss)
            
            # updates
            acc1, acc5 = Accuracy(logit_t, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            
            # backward
            loss.backward()
            optimizer.step()
            
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
        top1_test, top5_test, losses_test = H.validate(model=model_s, val_loader=val_loader,
                                                        criterion=criterion_cls, opt=opt)
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('test_loss', losses_test, epoch)
        writer.add_scalar('train_Acc', top1.avg, epoch)
        writer.add_scalar('train_top5_Acc', top5.avg, epoch)
        writer.add_scalar('test_Acc', top1_test, epoch)
        writer.add_scalar('test_top5_Acc', top5_test, epoch)
        if (x := top1_test) > best_acc:
            state = {
                'epoch': epoch,
                'opt': opt,
                'model': model_s.state_dict(),
                'acc': x,
                'best_acc': x,
                'optim': optimizer.state_dict(),
                'grt':grt,
                'loss':losses.avg
            }

            torch.save(state, f'{opt.save_folder}/student/{model_s._get_name()}_best.pth')
            best_acc = x
        with open(f'{opt.save_folder}/student/log.txt','a') as f:
            print(f'epoch: {epoch}, Best_Acc: {best_acc:.3f}, Acc@1: {top1.avg:.3f}, loss: {losses.avg:.3f}, Val_Acc: {top1_test}, Val_loss: {losses_test}, lr: {H.get_lr(optimizer)}',file= f)
        print(f'epoch: {epoch}, Best_Acc: {best_acc:.3f}, Acc@1: {top1.avg:.3f}, loss: {losses.avg:.3f}, Val_Acc: {top1_test}, Val_loss: {losses_test}, lr: {H.get_lr(optimizer)}')
        grt+=1
        if opt.scheduler =='cosineAnnealing':
            scheduler.step()
        
        state = {
                'epoch': epoch,
                'opt': opt,
                'model': model_s.state_dict(),
                'acc': top1_test,
                'best_acc':best_acc,
                'optim': optimizer.state_dict(),
                'grt':grt,
                'loss':losses.avg
            }

        torch.save(state, f'{opt.save_folder}/student/{model_s._get_name()}_curr.pth')
        # if epoch in save_epochs:
        #   torch.save(state, f'{opt.save_folder}/student/{model_s._get_name()}_{epoch}.pth')
    state = {
                'epoch': epoch,
                'opt': opt,
                'model': model_s.state_dict(),
                'acc': top1_test,
                'best_acc':best_acc,
                'optim': optimizer.state_dict(),
                'grt':grt,
                'loss':losses.avg
            }
    # state = {'opt': opt, 'model': model_s.state_dict()}
    torch.save(state, f'{opt.save_folder}/student/{model_s._get_name()}_last.pth')
                
 
    return top1.avg, losses.avg ##       
            
if __name__ == '__main__':
    main()