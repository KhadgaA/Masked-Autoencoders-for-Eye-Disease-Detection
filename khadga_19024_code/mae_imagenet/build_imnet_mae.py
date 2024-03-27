import sys
import os, torch
import numpy as np
sys.path.append('../mae_imagenet')
import mae_imagenet.models_mae as models_mae


import matplotlib.pyplot as plt
import torchvision.transforms as tt
sys.path.append('../data')
# from dataset.cifar10 import get_CIFAR10_dataloaders, get_CIFAR10_dataloaders_sample
# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
def prepare_model(chkpt_dir,opt, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model



def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()
    return im_paste

if __name__ == '__main__':
    pass
    # num_classes = 10
    # train_loader, val_loader, n_data = get_CIFAR10_dataloaders(batch_size=2, num_workers=2, is_instance=True)
    # chkpt_dir = 'mae_pretrain_vit_base_full.pth'#'mae_visualize_vit_large.pth'
    # model = prepare_model(chkpt_dir, 'mae_vit_base_patch16')#'mae_vit_large_patch16')
    # for data in train_loader:
    #     vitb_tf = tt.Compose([tt.Resize((224,224))])
    #     input, target, index = data
    #     inp = tt.Compose([tt.Resize([224,210])])(input)
    #     input = vitb_tf(inp)
    #     print(input.shape)
    #
    #     loss, y, mask = model(input.float(), mask_ratio=0.75)
    #     y = model.unpatchify(y)
    #     y = torch.einsum('nchw->nhwc', y).detach().cpu()
    #     plt.subplot(121)
    #     aug_input = y
    #     #plt.imshow((((aug_input[0] + 1) / 2).squeeze().permute(1,2,0)).detach().clamp(0,1).cpu())
    #     plt.imshow(torch.clip((y[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    #     plt.savefig(f'test')
    #     # plt.subplot(122)
    #     # plt.imshow((((input[0:5] + 1) / 2).squeeze().permute(1, 2, 0)).detach().clamp(0,1).cpu())
    #     # plt.show()
    #     break
