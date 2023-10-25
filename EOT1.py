import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import PIL
import numpy as np

import tempfile
from urllib.request import urlretrieve
import tarfile
import os

import json
import matplotlib.pyplot as plt


#load models
vgg = models.vgg16(pretrained=True).eval()
resnet50 = models.resnet50(pretrained=True).eval()

#load image and adjust the size
img_1 = PIL.Image.open("Figure_1.jpg")
img_1 = img_1.resize((224,224))
img_1 = (np.asarray(img_1) / 255.0).astype(np.float32)
# plt.imshow(img_1)
# plt.show()

#Demo function
#load labels
with open("image_net.json") as f:
    imagenet_labels = json.load(f)

def classify(img, correct_class = None, target_class = None, model = 'Vgg'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    
    #adjust the size of the input image and transform it to tensor datatype for model prediction
    tmp = torch.tensor(np.expand_dims(img.transpose((2,0,1)),axis = 0))
    
    ax1.imshow(img)
    fig.sca(ax1)
    
    if (model == 'Vgg'):
        p = nn.Softmax(dim = 1)(vgg(tmp)).detach().numpy()
    elif (model == 'Res'):
        p = nn.Softmax(dim = 1)(resnet50(tmp)).detach().numpy()
    else:
        raise("Unknown model")
    
    topk = list(np.argsort(p[0])[-10:])[::-1]
    topprobs = p[0][topk]
    
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()


#Tansfer blackbox attack(target model:vgg16 ; surrogate model:Resnet_50)
#target adversarial(PGD(I-FGSM))
alpha = 2/255
epsilon = 10/255
X_1 = torch.tensor(np.expand_dims(img_1.transpose((2,0,1)),axis = 0))
X_adv_1 = X_1.clone().detach()
for i in range(10):
    X_adv_1.requires_grad = True
    #target class: guacamole
    loss = nn.CrossEntropyLoss()(resnet50(X_adv_1),torch.tensor([292]).long())
    loss.backward()
    X_adv_1 = X_adv_1 + alpha*torch.sign(X_adv_1.grad)
    X_adv_1 = torch.min(X_adv_1, X_1+epsilon)
    X_adv_1 = torch.max(X_adv_1, X_1-epsilon)
    X_adv_1 = torch.clamp(X_adv_1, min=0, max=1)
    X_adv_1 = X_adv_1.detach()
adv_img_1 = X_adv_1[0].detach().numpy().transpose((1,2,0))
classify(img_1, correct_class = 292,model = 'Res')
classify(img_1, correct_class = 292,model = 'Vgg')
classify(adv_img_1,correct_class = 292,model = 'Res')
classify(adv_img_1,correct_class = 292,model = 'Vgg')

##
X_rot_adv_1 = T.functional.rotate(X_adv_1,angle = 22.5)
adv_rot_img_1 = X_rot_adv_1[0].detach().numpy().transpose((1,2,0))
classify(adv_rot_img_1,correct_class = 292,model = 'Vgg')
##

#Applying the EOT algorithm
alpha = 2/255
epsilon = 10/255
X_rot_1 = T.functional.rotate(X_1,angle = 22.5)
X_rot_adv_1 = X_rot_1.clone().detach()

for i in range(10):
    X_rot_adv_1.requires_grad = True
    loss = nn.CrossEntropyLoss()(resnet50(X_rot_adv_1),torch.tensor([292]).long())
    loss.backward()
    X_rot_adv_1 = X_rot_adv_1 + alpha*torch.sign(X_rot_adv_1.grad)
    X_rot_adv_1 = torch.min(X_rot_adv_1, X_rot_1 + epsilon)
    X_rot_adv_1 = torch.max(X_rot_adv_1, X_rot_1 - epsilon)
    X_rot_adv_1 = torch.clamp(X_rot_adv_1, min=0, max=1)
    X_rot_adv_1 = X_rot_adv_1.detach()
adv_rot_img_1 = X_rot_adv_1[0].detach().numpy().transpose((1,2,0))
classify(adv_rot_img_1,correct_class = 292,model = 'Res')
classify(adv_rot_img_1,correct_class = 292,model = 'Vgg')


#MI-FGSM
alpha = 1/255
g = 0
decay = 1
X_1 = torch.tensor(np.expand_dims(img_1.transpose((2,0,1)),axis = 0))
X_adv_1 = X_1.clone().detach()
for i in range(10):
    X_adv_1.requires_grad = True
    #target class: guacamole
    loss = nn.CrossEntropyLoss()(resnet50(X_adv_1),torch.tensor([292]).long())
    loss.backward()
    g = decay*g + X_adv_1.grad/torch.norm(X_adv_1.grad,p = 1)
    X_adv_1 = X_adv_1 + alpha*torch.sign(g)
    X_adv_1 = torch.clamp(X_adv_1, min=0, max=1)
    X_adv_1 = X_adv_1.detach()
adv_img_1 = X_adv_1[0].detach().numpy().transpose((1,2,0))
classify(img_1, correct_class = 292,model = 'Res')
classify(img_1, correct_class = 292,model = 'Vgg')
classify(adv_img_1,correct_class = 292,model = 'Res')
classify(adv_img_1,correct_class = 292,model = 'Vgg')

##
X_rot_adv_1 = T.functional.rotate(X_adv_1,angle = 22.5)
adv_rot_img_1 = X_rot_adv_1[0].detach().numpy().transpose((1,2,0))
classify(adv_rot_img_1,correct_class = 292,model = 'Vgg')
##

#Applying the EOT algorithm with MI-FGSM
alpha = 1/255
g = 0
decay = 1
X_rot_1 = T.functional.rotate(X_1,angle = 22.5)
X_rot_adv_1 = X_rot_1.clone().detach()

for i in range(10):
    X_rot_adv_1.requires_grad = True
    loss = nn.CrossEntropyLoss()(resnet50(X_rot_adv_1),torch.tensor([292]).long())
    loss.backward()
    g = decay*g + X_rot_adv_1.grad/torch.norm(X_rot_adv_1.grad,p = 1)
    X_rot_adv_1 = X_rot_adv_1 + alpha*torch.sign(g)
    X_rot_adv_1 = torch.clamp(X_rot_adv_1, min=0, max=1)
    X_rot_adv_1 = X_rot_adv_1.detach()
adv_rot_img_1 = X_rot_adv_1[0].detach().numpy().transpose((1,2,0))
classify(adv_rot_img_1,correct_class = 292,model = 'Res')
classify(adv_rot_img_1,correct_class = 292,model = 'Vgg')
