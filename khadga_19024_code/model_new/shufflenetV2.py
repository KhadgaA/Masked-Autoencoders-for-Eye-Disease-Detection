import torchvision.models as Models
import torch.nn as nn
import torch

class shuffle_modified(nn.Module):
    def __init__(self, original_model,num_classes,**kwargs) -> None:
        super(shuffle_modified,self).__init__()
        original_model = original_model(**kwargs)
        in_features = original_model.fc.in_features
        self.all_features = list(original_model.children())
        self.features = nn.Sequential(*self.all_features[:-1])
        self.latents = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(50,768)))
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.classifier = nn.Sequential(*list([nn.Flatten(),nn.Linear(in_features=in_features,out_features=num_classes)]))

        
    def forward(self,x,is_feat = False):
        x = self.features(x)
        l1 = self.latents(x)
        # l1 = x
        # print(l1.shape)
        l1 = torch.mean(l1,dim=1)
        l1 = self.norm(l1)
        x = x.mean([2, 3]) #global pool
        x = self.classifier(x)
        # x = self.classifier(x)
        if is_feat:
            return l1, x
        else:
            return x

def shufflenet_v2_x2_0(num_classes = 8,**kwargs):
    shuf_v2_x2_0 = Models.shufflenet_v2_x2_0
    return shuffle_modified(shuf_v2_x2_0,num_classes,**kwargs)

def shufflenet_v2_x1_0(num_classes = 8,**kwargs):
    shuf_v2_x1_0 = Models.shufflenet_v2_x1_0
    return shuffle_modified(shuf_v2_x1_0,num_classes,**kwargs)

if __name__ == '__main__':
    inp = torch.randn((64,3,224,224))
    print(shufflenet_v2_x2_0())
    # print(Models.shufflenet_v2_x2_0())
    out,_ = shufflenet_v2_x2_0()(inp)
    print(shufflenet_v2_x2_0(weights = Models.ShuffleNet_V2_X1_0_Weights),out.shape)