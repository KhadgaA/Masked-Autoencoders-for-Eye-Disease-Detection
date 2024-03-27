# This is the code for Knowledge Distillation, using a Masked Vision Transformer

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
from crd.criterion import CRDLoss
import helper as H
import utils as U
from model_new import *
import torchvision.models as Models
from torchvision import transforms
from torchvision import datasets


from mae_imagenet.build_imnet_mae import *
from data.ODIR import get_odir_dataloaders, get_odir_dataloaders_sample
from loss_funcs import *
from focal_loss import FocalLoss
from helper import Accuracy, AverageMeter


def augment(generator, images, opt):
    loss, predicted_img, mask, latents = generator(images, mask_ratio=opt.maskratio)
    predicted_img = generator.unpatchify(predicted_img)
    
    mask = mask.unsqueeze(-1).repeat(1, 1, generator.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = generator.unpatchify(mask)  # 1 is removing, 0 is keeping
    
    augment_image = images * (1 - mask) #+ predicted_img * mask  
    # augment_image = 0.8 * augment_image + 0.2 * images
    return augment_image.detach(), latents.detach()

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
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--cudaid', type=int, default=0, help='cuda device id')
   
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    

    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD',choices=['SGD', 'Adam'] ,help='optimizer to use, SGD or ADAM')
    parser.add_argument('--scheduler', type=str, default='multistep',choices=['multistep','cosineAnnealing'] ,help='choose a lr schedular')

    # dataset
    parser.add_argument('--dataset', type=str, default='ODIR', choices=['ODIR'], help='dataset')

    # generator
    parser.add_argument('--mvit', type=str, default='vitb', choices=['vitb'], help='the generator to use')#'vittiny'
    parser.add_argument('--maskratio', type=float, default= 0.75, help='masking ratio for Masked ViT')#.5
    
   
    parser.add_argument('--gen_epochs', type=int, default=10, help='how many epochs to train the generator for, each time')#20
    parser.add_argument('--train_epochs', type=int, default=25, help='how many training epochs for student before the generator training')#50
    # model
    parser.add_argument('--model_s', type=str, default='resnet18',
                        choices=['resnet50','resnet50_mod','resnet18','wrn_50_2','vgg16','densenet121','densenet201','shuv2_x1_0','shuv2_x2_0'])
    
    parser.add_argument('--model_t', type=str, default='resnet50', choices=['resnet50','resnet50_mod','resnet18','wrn_50_2','vgg16','densenet121','densenet201','shuv2_x1_0','shuv2_x2_0'], help='teacher model')
    parser.add_argument('--weights', type=str, default='odir', choices=['none','imagenet','odir'], help='teacher model')
    parser.add_argument('--weights_s', type=str, default='none', choices=['none','imagenet','odir'], help='student model')

     # use losses
    parser.add_argument('--crd', type=str, default='Flase', help='use crd loss', choices=['True', 'False'])
    parser.add_argument('--crd_gen', type=str, default='False', help='use crd loss wrt. Generator', choices=['True', 'False'])
    parser.add_argument('--use_gen', type = int, default=1, choices=[0,1], help='Use genertor for training(default = 0)')
    parser.add_argument('--use_teach', type = int, default=1, choices=[0,1], help='Use teacher for training(default = 1)')
    parser.add_argument('--use_hard_target', type = int, default=0, choices=[0,1], help='Use hard target label loss seperately for training(default = 1)')
    
    
    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4.0, help='temperature for KD distillation')#4
    parser.add_argument('--ce',type=str,default='ce',choices=['ce','focal_ce'],help='Crossentropy loss between student logits and target lables')
    parser.add_argument('--div',type=str,default='swd',choices=['kl','dkd', 'swd'],help='Distillation loss for true samples between student and teacher')
    parser.add_argument('--div_gen',type=str,default='cos',choices=['kl','mse','cos'], help= ' Distillation loss for augmented samples generated by generator, loss between student and teacher using augmented samples')# cos best
    parser.add_argument('--drop_beta',type=float,default=0.0, help='beta after drop_epoch')
    parser.add_argument('--drop_epoch',type=int,default=500, help='drop_epoch')
    
    #DKD
    parser.add_argument('-b', '--BETA', type=float, default=2.0, help=' beta param for dkd loss')
    parser.add_argument('-a', '--ALPHA', type=float, default=1.0, help='alpha param for dkd loss')

    
    # Auto Version folder
    parser.add_argument('--run_name', type=str, default= f'240e_swd_gen', help= 'name of the run (default "" )')
    parser.add_argument('--trial', type=str, default= '0', help= 'trial id of the run (default 0 : automatically creates new folder for each run,)')

     # NCE distillation
    parser.add_argument('--feat_dim', default=512, type=int, help='feature dimension')  # default = 128
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384//8, type=int,
                        help='number of negative samples for NCE')  # default 16384
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    opt = parser.parse_args()

    if opt.trial != '0':
        opt.model_path = f'./save/student_model/{opt.dataset},{opt.mvit},{opt.run_name},v{str(opt.trial)}'
        opt.tb_path = f"./save/student_tensorboards/{opt.dataset},{opt.mvit},{opt.run_name},v{str(opt.trial)}"
    else: 
        exists = True
        while exists:
            opt.trial = str(int(opt.trial) + 1)
            opt.model_path = f'./save/student_model/{opt.dataset},{opt.mvit},{opt.run_name},v{str(opt.trial)}'
            opt.tb_path = f"./save/student_tensorboards/{opt.dataset},{opt.mvit},{opt.run_name},v{str(opt.trial)}"
            exists = os.path.exists(opt.model_path)
             

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f'{opt.model_t}_{opt.model_s}_{opt.trial}'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    #
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
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
        if opt.crd == 'True':      
            train_loader, val_loader,n_data = get_odir_dataloaders_sample(batch_size=opt.batch_size,val_batch_size=opt.batch_size//2,
                                                            num_workers=opt.num_workers, k=opt.nce_k)
            print('using CRD')
        else:
            train_loader, val_loader,n_data = get_odir_dataloaders(batch_size=opt.batch_size,val_batch_size=opt.batch_size,
                                                            num_workers=opt.num_workers)
        num_classes = 8
        print('loaded ODIR')
        gen = './mae_imagenet/output_dir_mae_finetune/checkpoint-last.pth'
        
        if opt.model_t == 'vitb_clf':
            pass
        else: 
                       
            if opt.weights == 'none':
                model_t = model_dict[opt.model_t](num_classes=num_classes,weights = None)
            elif opt.weights == 'imagenet':
                model_t_weights = model_path_dict[opt.model_t + f'_{opt.weights}']
                model_t = model_dict[opt.model_t](num_classes=num_classes,weights =model_t_weights)
            elif opt.weights == 'odir':
                model_t_weights = model_path_dict[opt.model_t + f'_{opt.weights}']
                print(model_t_weights)
                model_t = model_dict[opt.model_t](num_classes=num_classes)
                model_t.load_state_dict(torch.load(model_t_weights)['model'])
            else:
                print('Weights not found/available')
                sys.exit()
        
            # model_s = model_dict[opt.model_s](num_classes=num_classes,weights = None if opt.weights == 'none'else model_path_dict[opt.model_s + f'_{opt.weights}']  )
            if opt.weights_s == 'none':
                model_s = model_dict[opt.model_s](num_classes=num_classes,weights = None)
            elif opt.weights_s == 'imagenet':
                model_s_weights = model_path_dict[opt.model_s + f'_{opt.weights_s}']
                print(model_s_weights)
                model_s = model_dict[opt.model_s](num_classes=num_classes,weights =model_s_weights)
            elif opt.weights_s == 'odir':
                model_s_weights = model_path_dict[opt.model_s + f'_{opt.weights_s}']
                print(model_s_weights)
                model_s = model_dict[opt.model_s](num_classes=num_classes)
                model_s.load_state_dict(torch.load(model_s_weights)['model'])
            else:
                print('Weights not found/available')
                sys.exit()
            # print(model_s)
            
            print('student params: ',numel(model_s,False))
            # sys.exit()
            print('teacher params: ',numel(model_t,False))

            
            
            if (opt.use_gen ==1) or (opt.crd_gen == 'True'):
                # generator model
                model_g = prepare_model(gen,opt,'mae_vit_base_patch16')
                # print(model_g)
                # print(,numel(model_g))
            # sys.exit()
    else:
        print('dataset not found')
        sys.exit()
    
    if opt.crd == 'True':
        data = torch.randn(2, 3, 224, 224)
        model_s.eval()
        feat_s, _ = model_s(data, is_feat=True)
        if opt.crd_gen == 'True':
            print('using CRD wrt. latents from generator there use_gen = True')
            model_g.eval()
            _, _, _, feat_t = model_g(data, mask_ratio=opt.maskratio)
        else:
            print('using CRD wrt. latents from teacher')
            model_t.eval()
            feat_t, _ = model_t(data, is_feat=True)
        print(feat_t.shape)
        
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    if opt.crd == 'True':
        f_s = torch.flatten(feat_s,1)
        f_t = torch.flatten(feat_t,1)
        # f_s = torch.cat([t.mean(1) for t in feat_s], dim=1)
        # f_t = torch.cat([t.mean(1) for t in feat_t], dim=1)
        
        # sys.exit()
        # print(f_t.shape)

        opt.s_dim = f_s.shape[-1]
        opt.t_dim = f_t.shape[-1]
        opt.n_data = n_data
        print(f_s.shape,f_t.shape,opt.s_dim,opt.t_dim,opt.n_data)
        # sys.exit()
        criterion_kd = CRDLoss(opt).to(device)
        print('using CRD Loss')
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

    
    if opt.use_hard_target == 1:
        if opt.ce =='ce':
            criterion_cls = nn.CrossEntropyLoss().to(device)
            print('using ce loss with hard targets')
        elif opt.ce == 'focal_ce':
            print('using focal_ce loss with hard targets')
            criterion_cls = FocalLoss(gamma=opt.gamma2).to(device)
        else:
            print('Loss not implemented')
            sys.exit()

    
    if opt.use_teach ==1:
        if opt.div =='kl':
            criterion_div = DistillKL(opt.kd_T).to(device)
            print('using KLDiv loss with teacher logits, use_hard_target = 1')
        elif opt.div == 'swd':
            criterion_div = SWD(alpha=1.0,beta=1.0).to(device)#alpha=3.0,beta=1.0 old kl3
            print('using Sample-Wise weighted loss with teacher logits, ce with target labels, therefore use_hard_target = 0')
        elif opt.div == 'dkd':
            criterion_div = DKD(opt).to(device)
            print('using dkd loss with teacher logits, use_hard_target = 1')
        else:
            print('Loss not implemented')
            sys.exit()
    
    if opt.use_gen ==1:
        if opt.div_gen =='kl':
            criterion_div_gen = DistillKL(opt.kd_T).to(device)
            print('using KLDiv loss with Generator latents')
        elif opt.div_gen =='mse':
            criterion_div_gen = nn.MSELoss().to(device)
            print('using MSE loss with Generator latents')
        elif opt.div_gen =='cos':
            criterion_div_gen = nn.CosineEmbeddingLoss().to(device)
            print('using Cosine loss with Generator latents')
        else:
            print('Loss not implemented')
            sys.exit()
    
     # optimizer
    if opt.optimizer =='SGD':
        optimizer = optim.SGD(trainable_list.parameters(), lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,nesterov=True)
    elif opt.optimizer =='Adam':
        optimizer = optim.Adam(trainable_list.parameters(), lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    if opt.scheduler =='cosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = opt.epochs)
        
    # append teacher after optimizer to avoid weight decay
    module_list.append(model_t)
    
    # generator
    if (opt.use_gen ==1) or (opt.crd_gen == 'True'):
        module_list.append(model_g)

    
    # Gpu check 
    if torch.cuda.is_available():
        module_list.to(device)
        # criterion_list.to(device)
        cudnn.benchmark = True
        
    val_loader = U.DeviceDataLoader(val_loader, device)
    result = H.evaluate(model_s, val_loader, nn.CrossEntropyLoss())
    print('student accuracy', result['val_acc'])
    
    e = 1 # starting epoch, handy when resuming training
    ## resume/load chkpt code.. 
    
    best_acc = 0 
    save_epochs = [50,100, 150, 200]
    print('traning start')
    for epoch in range(e, opt.epochs + 1):
        
        if epoch > opt.drop_epoch:
            opt.use_gen = 0
        # set module as train
        for module in module_list:
            module.train()
        model_s = module_list[0]
        # model_s.train()
        if (opt.use_gen ==1) or (opt.crd_gen == 'True'):
            model_t = module_list[-2]
            model_t.eval()
            model_g = module_list[-1]
            model_g.eval() 
        else:
            model_t = module_list[-1]
            model_t.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        if opt.scheduler =='multistep':
            H.adjust_learning_rate(epoch, opt, optimizer)
        writer.add_scalar('lr_student', H.get_lr(optimizer), epoch)
        # # print(grt,opt.train_epochs)
        # if grt == opt.train_epochs:
        #     grt = 0
        #     for ep in range(1,opt.gen_epochs+1):
        #             if opt.maxstp == 'True':
        #                 # Generator maximization step
        #                 # Nothing for now
        #                 pass
        #     tot +=1
        # schedule the loss_weights
        alpha1 = 1.0 #math.cos(epoch / 40) if epoch < 60 else 0.0
        alpha2 = 1.0 #1 - alpha1 if epoch < 60 else 1.0
               
        for data in tqdm(train_loader,desc = f'epoch: {epoch}/{opt.epochs} Stu'):
            # continue
            # ===================backward=====================
            optimizer.zero_grad()
            if opt.crd == 'True':
                input, target, index, contrast_idx = data
                contrast_idx = contrast_idx.to(device)
            else:
                input, target, index = data
            index = index.to(device)
            input = input.float()
            input = input.to(device,non_blocking=True)
            target = target.to(device,non_blocking=True)
            # Generator assisted latent space feature
            with torch.no_grad():
                # if epoch <= 50:
                if (opt.use_gen ==1) or (opt.crd_gen == 'True'):
                    aug_input, latents = augment(generator=model_g,images=input,opt = opt)
                    
                
                # print(latents.shape)
                # # print(aug_input.shape,latents.shape)
                # plt.subplot(121)
                # plt.imshow((((aug_input[0] + 1) / 2).squeeze().permute(1, 2, 0)).detach().clamp(0,1).cpu())
                # plt.subplot(122)
                # plt.imshow((((input[0] + 1) / 2).squeeze().permute(1, 2, 0)).detach().clamp(0,1).cpu())
                # plt.savefig(f'{opt.save_folder}/student/gen_{epoch}')
                #     # aug_input = aug_input
                if opt.use_teach ==1:
                    latent_t, logit_t = model_t(input, is_feat=True)
                    latent_t = latent_t.detach()
            with torch.cuda.amp.autocast():
                latent_s, logit_s = model_s(input,is_feat = True)           
                
                # losses
                loss_latents = 0.0
                loss_cls = 0.0
                loss_kl = 0.0
                loss_kd = 0.0
                loss = 0.0
                if opt.use_gen ==1:
                    latent_s = torch.flatten(latent_s,1)
                    latents = torch.flatten(latents,1)
                    loss_latents = criterion_div_gen(latent_s,latents,torch.ones(latents.shape[0]).to(device))
                if opt.use_teach ==1:
                    loss_kl = criterion_div(logit_s,logit_t,target)
                if opt.use_hard_target == 1:
                    loss_cls = criterion_cls(logit_s,target)
                if opt.crd == 'True':
                    # crd
                    l_s = torch.flatten(latent_s,1)
                    
                    
                    if opt.crd_gen == 'True':
                        l_t = torch.flatten(latents,1)
                    else:
                        l_t = torch.flatten(latent_t,1)
                    # print(l_s.shape,l_t.shape)
                    # sys.exit()
                    loss_kd = criterion_kd(l_s, l_t, index, contrast_idx)
                
                loss += alpha1 * loss_kl + alpha2 * loss_cls + loss_latents + loss_kd
                
            # print('loss',loss)
            
            # updates
            acc1, acc5 = Accuracy(logit_s, target, topk=(1, 5))
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
                                                        criterion=nn.CrossEntropyLoss(), opt=opt)
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
                # 'grt':grt,
                'loss':losses.avg
            }

            torch.save(state, f'{opt.save_folder}/student/{model_s._get_name()}_best.pth')
            best_acc = x
        with open(f'{opt.save_folder}/student/log.txt','a') as f:
            print(f'epoch: {epoch}, Best_Acc: {best_acc:.3f}, Acc@1: {top1.avg:.3f}, loss: {losses.avg:.3f}, Val_Acc: {top1_test}, Val_loss: {losses_test}, lr: {H.get_lr(optimizer)}',file= f)
        print(f'epoch: {epoch}, Best_Acc: {best_acc:.3f}, Acc@1: {top1.avg:.3f}, loss: {losses.avg:.3f}, Val_Acc: {top1_test}, Val_loss: {losses_test}, lr: {H.get_lr(optimizer)}')
        # grt+=1
        if opt.scheduler =='cosineAnnealing':
            scheduler.step()
        
        state = {
                'epoch': epoch,
                'opt': opt,
                'model': model_s.state_dict(),
                'acc': top1_test,
                'best_acc':best_acc,
                'optim': optimizer.state_dict(),
                # 'grt':grt,
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
                # 'grt':grt,
                'loss':losses.avg
            }
    # state = {'opt': opt, 'model': model_s.state_dict()}
    torch.save(state, f'{opt.save_folder}/student/{model_s._get_name()}_last.pth')
                
 
    return top1.avg, losses.avg ##       
            
if __name__ == '__main__':
    main()