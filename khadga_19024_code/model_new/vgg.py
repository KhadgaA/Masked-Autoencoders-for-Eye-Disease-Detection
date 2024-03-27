import torchvision.models as Models
import torch.nn as nn
import torch

class vgg_modified(nn.Module):
    def __init__(self, original_model,num_classes,**kwargs) -> None:
        super(vgg_modified,self).__init__()
        # original_model = Models.resnet18(num_classes=8)
        original_model = original_model(**kwargs)
        in_features = original_model.classifier[-1].in_features
        # out_features = original_model.classifier[-1].out_features
        self.features = nn.Sequential(*list(original_model.features.children()))
        
        self.latents = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(50,768)))
        self.norm = nn.LayerNorm(768, eps=1e-6)
        
        self.avgpool = nn.Sequential(nn.AdaptiveMaxPool2d((7,7)),nn.Flatten())
        self.classifier = nn.Sequential(*list(original_model.classifier.children()))
                                        # instead of ReLU at start now i'm using nn.LayerNorm since it was used in MViT also
        self.classifier[-1] = nn.Linear(in_features=in_features,out_features=num_classes)
       
    def forward(self,x,is_feat = False):
        x = self.features(x)
        l1 = self.latents(x)
        # l1 = x
        l1 = torch.mean(l1,dim=1)
        l1 = self.norm(l1)
        x = self.avgpool(x)
        x = self.classifier(x)
        # x = self.classifier(x)
        if is_feat:
            return l1, x
        else:
            return x
    
def vgg16(num_classes = 8,**kwargs):
    vgg16_odir = Models.vgg16
    
    return vgg_modified(vgg16_odir,num_classes,**kwargs)

if __name__ == '__main__':
    inp = torch.randn((64,3,224,224))
    print(vgg16())
    # print(Models.vgg16())
    out,_ = vgg16()(inp)
    print(vgg16(weights = Models.VGG16_Weights),out.shape)