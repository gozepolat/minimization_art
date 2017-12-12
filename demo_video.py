# -*- coding: utf-8 -*-
# Aydın Göze Polat, 2017
# Art with image regularization and PyTorch

import torch
from torch.autograd import Variable
from torch import squeeze
from torch import optim
from torch import nn
import cv2
import numpy as np

from modules.utils import image_to_variable, normalize, imshow, make_operator, imsave
from modules.regularizers import PeronaMalik


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier(m.bias.data)


def perona_malik_art(video_path=None):
    print("Perona-Malik on drugs...")

    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("camera capture device is not open..")
        exit(-1)

    res, image = cap.read()
    out = image_to_variable(image)
    out = normalize(out)
    pm = PeronaMalik(out.size()
                     , diffusion_rate=0.5
                     , delta_t=0.2
                     , coefficient='exp'
                     , learn_operator=True)
    pm.cuda()
    lr = 0.000000125
    optimizer = optim.SGD(pm.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0005)
    prev = None
    i = 0
    while True:
        res, image = cap.read()
        if not res:
            print("problem reading the image..")
            exit(-1)

        image = image_to_variable(image)
        image = normalize(image)
        out = pm.forward(out)

        # update the learning rate and refresh the operator
        out = Variable(out.data, requires_grad=False)
        if i % 8 != 0:
            out = out * 0.8 + image * 0.2
        else:
            i = 0
        optimizer.zero_grad()
        out = pm.forward(out)
        loss = torch.sum(torch.pow(pm.gradients*(out - image), 2) + pm.gradients + 10*torch.pow(out*(pm.gradients), 2))# + torch.pow(out[:,0]-out[:,2], 2) + torch.pow(out[:,0]-out[:,1], 2) + torch.pow(out[:,1]-out[:,2], 2))
        #if prev is not None:
        #    loss += 0.5 * torch.sum(torch.pow(out - prev, 2))
        prev = out
        loss.backward(retain_graph=True)
        optimizer.step()
        i += 1
        if i % 2 != 0:
            continue

        cv2.imshow("dodo", np.rollaxis(squeeze(out, 0).cpu().data.numpy(), 0, 3))
        cv2.waitKey(1)


if __name__ == "__main__":
    perona_malik_art()

