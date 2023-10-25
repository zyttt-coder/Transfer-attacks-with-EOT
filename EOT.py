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
inception = models.inception_v3(pretrained=True).eval()

#load image and adjust the size
img_path, _ = urlretrieve('https://github.com/lorenz-peter/lorenz-peter.github.io/raw/master/assets/img/cat.jpg')
img = PIL.Image.open(img_path)
img = img.resize((299,299))
img = (np.asarray(img) / 255.0).astype(np.float32)
# plt.imshow(img)
# plt.show()

#Demo function
#load labels
with open("image_net.json") as f:
    imagenet_labels = json.load(f)

def classify(img, correct_class = None, target_class = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    
    #adjust the size of the input image and transform it to tensor datatype for model prediction
    tmp = torch.tensor(np.expand_dims(img.transpose((2,0,1)),axis = 0))
    
    ax1.imshow(img)
    fig.sca(ax1)
    
    p = nn.Softmax(dim = 1)(inception(tmp)).detach().numpy() #transform the tensor(Softmax result) back to numpy for argsort
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

#target adversarial(PGD)
#PGD step size
alpha = 2/255
#Maximum allowable pertubation
epsilon = 8/255
X = torch.tensor(np.expand_dims(img.transpose((2,0,1)),axis = 0))
X_adv = X.clone().detach()
for i in range(10):
    X_adv.requires_grad = True
    #target class: guacamole
    loss = nn.CrossEntropyLoss()(inception(X_adv),torch.tensor([924]).long())
    loss.backward()
    X_adv = X_adv - alpha*torch.sign(X_adv.grad)
    X_adv = torch.min(X_adv, X+epsilon)
    X_adv = torch.max(X_adv, X-epsilon)
    X_adv = torch.clamp(X_adv, min=0, max=1)
    X_adv = X_adv.detach()
adv_img = X_adv[0].detach().numpy().transpose((1,2,0))

classify(img, correct_class = 281)
classify(adv_img,correct_class = 281,target_class = 924)

#EOT
#First check whether the adversarial example is still adversarial after the rotation
X_rot_adv = T.functional.rotate(X_adv,angle = 22.5)
adv_rot_img = X_rot_adv[0].detach().numpy().transpose((1,2,0))
classify(adv_rot_img,correct_class = 281,target_class = 924)

#Applying the EOT algorithm
alpha = 2/255
epsilon = 8/255
X_rot = T.functional.rotate(X,angle = 22.5)
X_rot_adv = X_rot.clone().detach()

for i in range(10):
    X_rot_adv.requires_grad = True
    loss = nn.CrossEntropyLoss()(inception(X_rot_adv),torch.tensor([924]).long())
    loss.backward()
    X_rot_adv = X_rot_adv - alpha*torch.sign(X_rot_adv.grad)
    X_rot_adv = torch.min(X_rot_adv, X_rot + epsilon)
    X_rot_adv = torch.max(X_rot_adv, X_rot - epsilon)
    X_rot_adv = torch.clamp(X_rot_adv, min=0, max=1)
    X_rot_adv = X_rot_adv.detach()
adv_rot_img = X_rot_adv[0].detach().numpy().transpose((1,2,0))
classify(adv_rot_img,correct_class = 281,target_class = 924)










