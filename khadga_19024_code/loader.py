import torch
import torchvision
import torchvision.datasets as dts
import torchvision.transforms as tt
import utils as U
from torch.utils.data import DataLoader as DL






def dataloader(dataset ='cifar10',test:bool = True,train_trnsforms=None,test_trnsforms = None,device = U.get_default_device(),batch_size = 200,root = './data'):
    # dataset = 'cifar10'
    data_avail = False
    for x in dts.__all__:
        if str(dataset).lower() == x.lower():
            data_avail = True
            dataset = x
            print(x)
            break
    train_trnsform = [tt.ToTensor()]
    if train_trnsforms is not None:
        train_trnsform =  train_trnsforms
    train_trnsform = tt.Compose(train_trnsform)



    Dataset = getattr(dts,dataset)

    train_dataset = Dataset(root=root, train=True, download=True, transform=train_trnsform)
    train_loader = DL(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1,pin_memory=True)
    train_dl = U.DeviceDataLoader(train_loader, device)

    if test:
        test_trnsform = [tt.ToTensor()]
        if test_trnsforms is not None:
            test_trnsform = test_trnsforms
        test_trnsform = tt.Compose(test_trnsform)

        test_dataset = Dataset(root=root, train=False, download=True, transform=test_trnsform)
        test_loader = DL(test_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)
        test_dl = U.DeviceDataLoader(test_loader, device)
        return train_dl, test_dl
    return train_dl





def test_loader(test_trnsforms=None,dataset=None,train=False,batch_size = 200,device = U.get_default_device(),root='./data'):
    data_avail = False
    for x in dts.__all__:
        if str(dataset).lower() == x.lower():
            data_avail = True
            dataset = x
            print(x)
            break

    test_trnsform = [tt.ToTensor()]
    if test_trnsforms is not None:
        test_trnsform = test_trnsforms
    test_trnsform = tt.Compose(test_trnsform)

    Dataset = getattr(dts,dataset)

    test_dataset = Dataset(root=root, train=train, download=True, transform=test_trnsform)
    test_loader = DL(test_dataset,batch_size=batch_size,shuffle=False,num_workers=1,pin_memory=True)

    test_dl = U.DeviceDataLoader(test_loader, device)
    return test_dl