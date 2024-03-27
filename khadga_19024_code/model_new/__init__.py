from .resnets import Resnet18, Resnet50, Resnet50_mod
from .wrn import wide_resnet50_2
from .vgg import vgg16
from .densenet import densenet121,densenet201
from .shufflenetV2 import shufflenet_v2_x1_0, shufflenet_v2_x2_0
import torchvision.models as Models
import os
model_dict = {
    'resnet50': Resnet50,
    'resnet50_mod': Resnet50_mod,
    'resnet18':Resnet18,
    'wrn_50_2': wide_resnet50_2,
    'vgg16': vgg16,
    'densenet121': densenet121,
    'densenet201': densenet201,
    'shuv2_x1_0': shufflenet_v2_x1_0,
    'shuv2_x2_0': shufflenet_v2_x2_0,
}

model_path_dict = {
    'resnet50_mod_odir': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../save/teacher_model/ODIR,v1/resnet56_1/student/ResNet_best.pth'),
    'resnet50_odir':os.path.join(os.path.dirname(os.path.abspath(__file__)),'../save/teacher_model/ODIR,resnet50,finetune50e,v1/resnet50_1/student/resnet_modified_best.pth'),
    'resnet18_odir':os.path.join(os.path.dirname(os.path.abspath(__file__)),'../save/teacher_model/ODIR,resnet18,finetune50e,v1/resnet18_1/student/resnet_modified_best.pth'),
    'wrn_50_2_odir': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../save/teacher_model/ODIR,wrn_50_2,finetune50e,v2/wrn_50_2_2/student/wrn_modified_best.pth'),
    'vgg16_odir': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../save/teacher_model/ODIR,vgg16,finetune50e,v2/vgg16_2/student/vgg_modified_best.pth'),
    'densenet121_odir':os.path.join(os.path.dirname(os.path.abspath(__file__)), '../save/teacher_model/ODIR,densenet121,finetune50e,v1/densenet121_1/student/densenet_modified_best.pth'),
    'densenet201_odir': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../save/teacher_model/ODIR,densenet201,finetune50e,v1/densenet201_1/student/densenet_modified_best.pth'),
    'shuv2_x1_0_odir':os.path.join(os.path.dirname(os.path.abspath(__file__)), '../save/teacher_model/ODIR,shuv2_x1_0,finetune50e,v1/shuv2_x1_0_1/student/shuffle_modified_best.pth'),
    'shuv2_x2_0_odir':os.path.join(os.path.dirname(os.path.abspath(__file__)), '../save/teacher_model/ODIR,shuv2_x2_0,finetune50e,v1/shuv2_x2_0_1/student/shuffle_modified_best.pth'),
    
    'resnet50_imagenet': Models.ResNet50_Weights,
    'resnet50_mod_imagenet': Models.ResNet50_Weights,
    'resnet18_imagenet':Models.ResNet18_Weights,
    'wrn_50_2_imagenet': Models.Wide_ResNet50_2_Weights,
    'vgg16_imagenet': Models.VGG16_Weights,
    'densenet121_imagenet': Models.DenseNet121_Weights,
    'densenet201_imagenet': Models.DenseNet201_Weights,
    'shuv2_x1_0_imagenet': Models.ShuffleNet_V2_X1_0_Weights,
    'shuv2_x2_0_imagenet': Models.ShuffleNet_V2_X2_0_Weights,
}
