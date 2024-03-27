import torchvision.models as Models
import torch.nn as nn
import torch

class resnet_modified(nn.Module):
    def __init__(self, original_model,num_classes,**kwargs) -> None:
        super(resnet_modified,self).__init__()
        original_model = original_model(**kwargs)
        in_features = original_model.fc.in_features
        # out_features = original_model.fc.out_features
        self.all_features = list(original_model.children())
        self.features = nn.Sequential(*self.all_features[:-2])
        # nn.Sequential(*list(original_model.get_feat_modules()))
        self.latents = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(50,768)))
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.classifier = nn.Sequential(*list([nn.AdaptiveAvgPool2d(output_size=(1,1)), nn.Flatten(),nn.Linear(in_features=in_features,out_features=num_classes)]))
        # print(original_model.avgpool)
        # self.classifier[-1].out_features = num_classes
    def forward(self,x,is_feat=False):
        x = self.features(x)
        # print(x.shape)
        # l1 = x
        l1 = self.latents(x)
        # l1 = x
        # print(l1.shape,x.shape)
        l1 = torch.mean(l1,dim=1)
        l1 = self.norm(l1)
        
        # print(x.shape)
        x = self.classifier(x)
        # x = self.classifier(x)
        if is_feat:
            return l1, x
        else:
            return x

def Resnet50_mod(num_classes = 8,**kwargs):
    Resnet50_odir = Models.resnet50(**kwargs)
    # # Freeze weights
    # for param in model_t.parameters():
    #     param.requires_grad = False
    in_features = Resnet50_odir.fc.in_features

    # Defining Dense top layers after the convolutional layers
    Resnet50_odir.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=in_features),    
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes))
    return Resnet50_odir

def Resnet18(num_classes = 8,**kwargs):
    resnet18_odir = Models.resnet18
    return resnet_modified(resnet18_odir,num_classes)
def Resnet50(num_classes = 8,**kwargs):
    resnet50_odir = Models.resnet50
    return resnet_modified(resnet50_odir,num_classes,**kwargs)

if __name__ == '__main__':
    inp = torch.randn((64,3,224,224))
    out,_ = Resnet50()(inp,is_feat = True)
    print(Resnet50(weights = Models.ResNet50_Weights),out.shape)