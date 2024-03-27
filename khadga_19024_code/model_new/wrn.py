import torchvision.models as Models
import torch.nn as nn
import torch

class wrn_modified(nn.Module):
    def __init__(self, original_model,num_classes,**kwargs) -> None:
        super(wrn_modified,self).__init__()
        original_model = original_model(**kwargs)
        # original_model = Models.resnet18(num_classes=8)
        in_features = original_model.fc.in_features
        out_features = original_model.fc.out_features
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        # nn.Sequential(*list(original_model.get_feat_modules()))
        self.latents = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(50,768)))
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.Flatten(), nn.Linear(in_features=in_features,out_features=num_classes))# instead of ReLU at start now i'm using nn.LayerNorm since it was used in MViT also
        # self.classifier = nn.Sequential(nn.Linear(in_features=512,out_features=8))
        # self.classifier = nn.Linear(in_features=768, out_features=8)
        
    def forward(self,x,is_feat=False):
        x = self.features(x)
        l1 = x
        l1 = self.latents(l1)
        # l1 = x
        # print(l1.shape)
        l1 = torch.mean(l1,dim=1)
        l1 = self.norm(l1)
        x = self.classifier(x)
        # x = self.classifier(x)
        if is_feat:
            return l1, x
        else:
            return x
        
def wide_resnet50_2(num_classes = 8,**kwargs):
    wide_resnet50_2 = Models.wide_resnet50_2
    return wrn_modified(wide_resnet50_2,num_classes)

if __name__ == '__main__':
    inp = torch.randn((64,3,224,224))
    print(wide_resnet50_2())
    # print(Models.vgg16())
    out,_ = wide_resnet50_2()(inp)
    print(wide_resnet50_2(weights = Models.Wide_ResNet50_2_Weights),out.shape)