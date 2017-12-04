# -*- coding: utf-8 -*-
# Aydın Göze Polat, 2017
# Art with image regularization and Pytorch

import torch
from torch.autograd import Variable
from torch import squeeze
from torch import optim
from torch import nn

from modules.utils import image_to_variable, normalize, imshow, make_operator
from modules.regularizers import PeronaMalik


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier(m.bias.data)


def perona_malik_art(image):
    print("Perona-Malik model...")
    image = image_to_variable(image)
    pm = PeronaMalik(image.size()
                     , diffusion_rate=0.2
                     , delta_t=0.2
                     , coefficient='exp'
                     , learn_operator=True)

    lr = 0.000002
    optimizer = optim.SGD(pm.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0005)
    pm.cuda()
    original = normalize(image)

    out = pm.forward(original)
    for i in range(100001):
        optimizer.zero_grad()
        out = Variable(out.data, requires_grad=False)
        out = pm.forward(out)
        loss = torch.sum(torch.pow((out - original),2) + torch.abs(pm.gradients) + torch.pow(out,2))
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 75 == 0:
            print(loss)
            print("iteration %d" % i)
            imshow(squeeze(squeeze(out, 0), 0).cpu().data.numpy())

            # update the learning rate and refresh the operator
            if i % 375 == 300:
                pm.laplace = make_operator('laplace', requires_grad=True)
                lr *= 0.995
                optimizer = optim.SGD(pm.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0005)
                out = out * 0.6 + original * 0.4


def perona_malik_vanilla(image):
    print("Perona-Malik model...")
    image = image_to_variable(image)

    pm = PeronaMalik(image.size()
                     , diffusion_rate=0.2
                     , delta_t=0.01
                     , coefficient='exp')
    pm.cuda()
    original = normalize(image)
    out = pm.forward(original)

    for i in range(5001):
        pm.zero_grad()
        out = pm.forward(Variable(out.data))
        if i % 100 == 0:
            print("iteration %d" % i)
            imshow(squeeze(squeeze(out, 0), 0).cpu().data.numpy())


if __name__ == "__main__":
    input_image = 'images/star.jpg'
    perona_malik_art(input_image)

