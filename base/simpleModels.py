"""
Defined functions for general classification model building
Essentially the functions from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet
from pretrainedmodels import se_resnext101_32x4d, inceptionresnetv2
from efficientnet_pytorch import EfficientNet

from .simpleLosses import create_loss_fx
from .simpleOptimizer import create_optimizer

def load_model(opt, model_ft):
    # input
    model_ft = model_ft.cuda()
    if opt.prev_model != None and not opt.joint_fusion and not opt.parallel_model:
        model_ft = torch.load(opt.prev_model).cuda()
    else:
        model_ft = model_ft

    ### NEED SEPERATE SCRIPTS FOR OPTIMIZERS AND LOSS FUNCTIONS ###

    # Observe that all parameters are being optimized
    optimizer_ft = create_optimizer(opt)

    # Setup the loss fxn
    if opt.parallel_model:
        opt.t1_cls_weights = torch.tensor(opt.t1_cls_weights).cuda()
        opt.t2_cls_weights = torch.tensor(opt.t2_cls_weights).cuda()
        
        criterion1 = nn.CrossEntropyLoss(weight=opt.t1_cls_weights)
        criterion2 = nn.CrossEntropyLoss(weight=opt.t2_cls_weights)
        
        return model_ft, optimizer_ft, criterion1, criterion2
    else:
        criterion = create_loss_fx(opt)

        return model_ft, optimizer_ft, criterion

def set_parameter_requires_grad(model_ft, model_freeze, parallel=False, freeze_backbone=False):
    """
    function to set the parameters to be trained based on the model_freeze
    model_freeze (float):
        - '1.0': freeze the whole model, only extract features
        - '0.0': do not freeze the model at all, train the full model parameters
        - '0.X': freeze the first fraction (0.X) of the model parameters
    """
    params_to_update = []
    if parallel:
        if freeze_backbone:
            for param in model_ft.backbone.parameters():
                param.requires_grad = False
            for name, param in model_ft.task1_clf.named_parameters():
                params_to_update.append(param)
            for name, param in model_ft.task2_clf.named_parameters():
                params_to_update.append(param)
        else:
            for name, param in model_ft.named_parameters():
                params_to_update.append(param)

        return params_to_update
    else:
        if model_freeze == 1.0:
            for param in model_ft.parameters():
                param.requires_grad = False
        else:
            params_to_update = []
            num_layers = len([name for name, param in model_ft.named_parameters()])
            frozen_layers = round(num_layers * model_freeze)
            if frozen_layers != 0 and frozen_layers % 2 != 0:  # rounds up to even number for weight and bias combinations
                frozen_layers += 1
            layer_count = 0
            for name, param in model_ft.named_parameters():
                if layer_count < frozen_layers:
                    param.requires_grad = False
                else:
                    params_to_update.append(param)
                layer_count += 1
        return params_to_update

class simpleConvNet(nn.Module):
    def __init__(self, cls_out, conv_layers=[6, 16], kernel=5, clf_layers=[120, 84]):
        """
        input: 
            conv_layers: convolution layers to add e.g. [6, 16, 32] is 3 convolution layers with n_channels -> 6 -> 16 -> 32
            kernel: kernel size to use e.g. [5] is apply kernel size 5x5 for all convolutions
            clf_layers: linear classification layers to add e.g. [120, 84, 2] means linear layer going from input to 120, to 84 to 2 
        """
        super().__init__()
        # make convolution layers
        self.convolutions = torch.nn.Sequential()
        for conv in range(len(conv_layers)):
            self.convolutions.add_module('conv_{}'.format(conv+1), nn.Conv2d(conv_layers[conv - 1], conv_layers[conv], kernel) if conv != 0 else nn.Conv2d(3, conv_layers[conv], kernel))
            self.convolutions.add_module('relu_{}'.format(conv+1), torch.nn.ReLU())
            self.convolutions.add_module('maxpool_{}'.format(conv+1), torch.nn.MaxPool2d(2, 2))
        
        # make linear classification layers
        self.classification = torch.nn.Sequential()
        for linear in range(len(clf_layers)):
            self.classification.add_module('linear_{}'.format(linear + 1), nn.Linear(clf_layers[linear - 1], clf_layers[linear]) if linear != 0 else nn.Linear(conv_layers[-1] * kernel * kernel, clf_layers[linear]))
            self.classification.add_module('relu_{}'.format(linear+conv+1), torch.nn.ReLU())
        self.classification.add_module('final_linear', nn.Linear(clf_layers[linear], cls_out))
    
    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.classification(x)
        return x


class jointFusionCNN(nn.Module):
    def __init__(self, opt, model, img_num_ftrs):
        """
        input:
            opt: base_options
                Need: opt.n_EHR - number of EHR features being used to fuse
            model: backbone model to use as image branch
            n_ftrs: number of features from the image branch
        output:
            joint fusion model
        """
        super().__init__()
        # make EHR branch with dense layers
        self.EHR_branch = nn.Sequential(
            nn.Linear(opt.n_EHR, opt.n_EHR, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        # make image branch with the loaded model
        self.Image_branch = torch.nn.Sequential()
        self.Image_branch.add_module('Backbone', model)
        i = 0
        while img_num_ftrs > (opt.n_EHR * 2):
            self.Image_branch.add_module('Linear_{}'.format(i),  nn.Linear(img_num_ftrs, img_num_ftrs // 2, bias=False))
            self.Image_branch.add_module('ReLU_{}'.format(i), nn.LeakyReLU(inplace=True))
            img_num_ftrs = img_num_ftrs // 2
            i += 1
        self.Image_branch.add_module('Linear_{}'.format(i), nn.Linear(img_num_ftrs, opt.n_EHR, bias=False))
        self.Image_branch.add_module('ReLU_{}'.format(i), nn.LeakyReLU(inplace=True))
        # make classification "main"
        self.fc = nn.Sequential(
            nn.Linear(opt.n_EHR * 2, opt.num_classes, bias=False),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, img, ehr):
        # get image features
        img_ft = self.Image_branch(img)
        # get EHR features
        ehr_ft = self.EHR_branch(ehr)
        # concat features
        joint_ft = torch.cat((img_ft, ehr_ft), 1)
        # classify
        return self.fc(joint_ft)


class ParallelNetwork(nn.Module):
    def __init__(self, opt, backbone_model, num_ftrs):
        """
        input:
            opt: base_options
            backbone_model: backbone model to use to extract the features (for now just resnext101)
            task1_cls: number of classes in the first task classificaiton
            task2_cls: number of classes in the first task classificaiton
        output:
            task1_loss
            task2_loss
        """
        super().__init__()
        # remove last layer and use Identity to get features from it
        self.backbone = backbone_model
        self.backbone.fc = nn.Identity()

        # make classification layer for task1
        self.task1_clf = nn.Sequential(
            nn.Linear(num_ftrs, opt.task1_cls, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        # make classification layer for task2
        self.task2_clf = nn.Sequential(
            nn.Linear(num_ftrs, opt.task2_cls, bias=False),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, img):
        # get image features from backbone
        img_fts = self.backbone(img)
        # get task1 output
        task1_out = self.task1_clf(img_fts)
        # get task1 output
        task2_out = self.task2_clf(img_fts)
        return task1_out, task2_out


def initialize_model(opt):
    """
    Requires: opt.model_name, opt.use_pretrained, opt.model_freeze, opt.num_classes

    To add another model into this function:
    1) install the model; 2) create a model_name; 3) copy the format of the other models and create the
    model_ft, set_parameter, etc. 3.5) the num_ftrs might need to load different names (i.e. model_ft.fc or
    model_ft._fc - different models will have different final layers but will generally be 'fc' for pytorch pretrained
    models; '_fc' for efficientNet and 'last_layer' for pretrainedmodels)
    4) add the final layer as just a linear layer*
    4.5)* final layer needs to be a linear layer because pytorch loss functions include softmax and sigmoid activations
    within them
    5) make sure to read the originial architecture and set the input size

    Currently the models I have added are:
    EfficientNet-B7, Resnet18, Resnet50, Resnet101, Resnet152, Alexnet, VGG11_bn, VGG19_bn, Squeezenet1.1, Densenet121,
    Densenet169, Densenet161, MobileNetV2, and Inception v3

    see link for more models and details on each: https://pytorch.org/docs/stable/torchvision/models.html
    """
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = None
    is_inception = False
    
    if opt.joint_fusion:
        print('making joint fusion model')
    
    if opt.model_name == 'simpleConvNet':
        model_ft = simpleConvNet(opt.num_classes)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        input_size = 32

    elif opt.model_name == "resnext101":
        """resnext101_32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=opt.use_pretrained)
        #params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.fc.in_features
        if opt.joint_fusion:
            print('making joint fusion model')
            if opt.prev_model is not None:
                model_ft = torch.load(opt.prev_model)
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
            params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        elif opt.parallel_model:
            print('making parallel model')
            if opt.prev_model is not None:
                model_ft = torch.load(opt.prev_model)
            model_ft = ParallelNetwork(opt, model_ft, num_ftrs)
            params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze, parallel=True, freeze_backbone=opt.freeze_backbone)
        else:
            params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
            model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "se-resnext101":
        """ SE-ResNeXt101
        """
        model_ft = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet') if opt.use_pretrained else se_resnext101_32x4d(num_classes=1000, pretrained=None)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.last_linear.in_features
        if opt.joint_fusion:
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.last_linear = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224
    
    elif opt.model_name == "inceptionresnetv2":
        """ Inception ResNet v2
        """
        model_ft = inceptionresnetv2(num_classes=1000, pretrained='imagenet') if opt.use_pretrained else inceptionresnetv2(num_classes=1000, pretrained=None)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.last_linear.in_features
        if opt.joint_fusion:
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.last_linear = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 299
        is_inception = True

    elif 'efficientnet' in opt.model_name:
        """ EfficientNet-BX (model name should be: 'efficientnet-bX', where X=0-7)
        """
        model_ft = EfficientNet.from_pretrained(opt.model_name) if opt.use_pretrained else EfficientNet.from_name(opt.model_name)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft._fc.in_features
        if opt.joint_fusion:
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif 'resnet' in opt.model_name:
        if opt.model_name == 'resnet9':
            """ Modified Resnet18 with only one block per layer so that it's 'half'
            - this is a good way to make custom models from pytorch classes
            """
            model_ft = resnet._resnet('resnet18_half', resnet.BasicBlock, [1, 1, 1, 1], False, False)
        elif opt.model_name == "resnet18":
            """ Resnet18 """
            model_ft = models.resnet18(pretrained=opt.use_pretrained)
        elif opt.model_name == "resnet50":
            """ Resnet50 """
            model_ft = models.resnet50(pretrained=opt.use_pretrained)
        elif opt.model_name == "resnet101":
            """ Resnet101 """
            model_ft = models.resnet101(pretrained=opt.use_pretrained)
        elif opt.model_name == "resnet152":
            """ Resnet152 """
            model_ft = models.resnet152(pretrained=opt.use_pretrained)
        else:
            print('{} is not a supported ResNet model'.format(opt.model_name))
            return model_ft, input_size

        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.fc.in_features
        if opt.joint_fusion:
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "alexnet":
        """ Alexnet """
        model_ft = models.alexnet(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.classifier[6].in_features
        if opt.joint_fusion:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif 'vgg' in opt.model_name:
        if opt.model_name == "vgg11bn":
            """ VGG11_bn """
            model_ft = models.vgg11_bn(pretrained=opt.use_pretrained)
        elif opt.model_name == "vgg19bn":
            """ VGG19_bn """
            model_ft = models.vgg19_bn(pretrained=opt.use_pretrained)
        else:
            print('{} is not a supported VGG model'.format(opt.model_name))
            return model_ft, input_size
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.classifier[6].in_features
        if opt.joint_fusion:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "squeezenet1.1":
        """ Squeezenet1.1 """
        model_ft = models.squeezenet1_1(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        if opt.joint_fusion:
            model_ft.classifier[1] = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = 512
            model_ft = jointFusionCNN(opt, model_ft, 512)
        else:
            model_ft.classifier[1] = nn.Conv2d(512, opt.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = opt.num_classes
        input_size = 224

    elif 'densenet' in opt.model_name:
        if opt.model_name == "densenet121":
            """ Densenet121"""
            model_ft = models.densenet121(pretrained=opt.use_pretrained)
        elif opt.model_name == "densenet161":
            """ Densenet161"""
            model_ft = models.densenet161(pretrained=opt.use_pretrained)
        elif opt.model_name == "densenet169":
            """ Densenet169"""
            model_ft = models.densenet169(pretrained=opt.use_pretrained)
        else:
            print('{} is not a supported DenseNet model'.format(opt.model_name))
            return model_ft, input_size
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)	
        num_ftrs = model_ft.classifier.in_features
        if opt.joint_fusion:
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "mobilenet":
        """ MobileNetV2
        """
        model_ft = models.mobilenet_v2(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        num_ftrs = model_ft.classifier[1].in_features
        if opt.joint_fusion:
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = jointFusionCNN(opt, model_ft, num_ftrs)
        else:
            model_ft.classifier[1] = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 224

    elif opt.model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=opt.use_pretrained)
        params_to_update = set_parameter_requires_grad(model_ft, opt.model_freeze)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
        input_size = 299
        is_inception = True

    else:
        print("Invalid model name")

    return model_ft, params_to_update, input_size, is_inception