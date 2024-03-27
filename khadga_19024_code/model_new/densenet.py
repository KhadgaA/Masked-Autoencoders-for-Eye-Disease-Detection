import torchvision.models as Models
import torch.nn as nn
import torch.nn.functional as F
import torch

class densenet_modified(nn.Module):
    def __init__(self, original_model,num_classes,**kwargs) -> None:
        super(densenet_modified,self).__init__()
        original_model = original_model(**kwargs)
        in_features = original_model.classifier.in_features
        out_features = original_model.classifier.out_features
        self.all_features = list(original_model.children())[:-1]
        self.features = nn.Sequential(*self.all_features[:-1])
        self.norm5 = self.all_features[-1]
        self.latents = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(50,768)))
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.classifier = nn.Sequential(*list([nn.Flatten(),nn.Linear(in_features=in_features,out_features=num_classes)]))

    def forward(self,x,is_feat = False):
        x = self.features(x)
        l1 = self.latents(x)
        x = self.norm5(x)
        # l1 = x
        # print(l1.shape)
        l1 = torch.mean(l1,dim=1)
        l1 = self.norm(l1)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = self.classifier(x)
        if is_feat:
            return l1, x
        else:
            return x
def densenet121(num_classes = 8,**kwargs):
    densnet121 = Models.densenet121
    return densenet_modified(densnet121,num_classes,**kwargs)

def densenet201(num_classes = 8,**kwargs):
    densnet201 = Models.densenet201
    return densenet_modified(densnet201,num_classes,**kwargs)
if __name__ == '__main__':
    inp = torch.randn((64,3,224,224))
    print(densenet201())
    print(Models.densenet201())
    out,_ = densenet201()(inp)
    print(densenet201(weights = Models.DenseNet201_Weights),out.shape)