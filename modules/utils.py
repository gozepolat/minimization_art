# -*- coding: utf-8 -*-
# Aydın Göze Polat, 2017

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from PIL import Image


def imload(impath):
    image = np.rollaxis(np.asarray(Image.open(impath)), 2, 0)
    return torch.unsqueeze(cuda_cast(torch.from_numpy(image)), 0)


def image_to_variable(image, requires_grad=False):
    if isinstance(image, basestring):
        image = imload(image)
    return Variable(image, requires_grad=requires_grad)


def scalar_to_parameter(value, size, requires_grad=False, dtype=torch.FloatTensor):
    return Parameter(dtype([value]).cuda(), requires_grad=requires_grad).expand(size)


def normalize(image):
    image = image - torch.min(image).expand_as(image)
    image = image / torch.max(image).expand_as(image)
    return image


def imshow(img, duration=0.001):
    plt.imshow(np.rollaxis(img, 0, 3))
    plt.pause(duration)


def imsave(path, img):
    plt.imsave(path, np.rollaxis(img, 0, 3))


def get_np_operator(name, in_channels, out_channels):
    if name == 'laplace':
        return np.array([[[[0.0, 1.0, 0.0]
                         , [1.0, -4.0, 1.0]
                         , [0.0, 1.0, 0.0]]]
                         * in_channels] * out_channels)
    if name == 'scharr_x':
        return np.array([[[[3.0, 0.0, -3.0]
                         , [10.0, 0.0, -10.0]
                         , [3.0, 0.0, -3.0]]]
                         * in_channels] * out_channels)
    if name == 'scharr_y':
        return np.array([[[[3.0, 10.0, 3.0]
                         , [0.0, 0.0, 0.0]
                         , [-3.0, -10.0, -3.0]]]
                         * in_channels] * out_channels)
    raise Exception("Name %s not found"%name)


def make_operator(name, in_channels=3, out_channels=3, requires_grad=False):
    """create a gradient operator for edge detection and image regularization tasks

    :param name: laplace, scharr_x, or scharr_y
    :param in_channels: e.g. 3 for an RGB image
    :param out_channels: usually the same as the # input channels
    :param requires_grad: learnable operator
    :return: convolutional kernel initialized with one of the operator values
    """
    dtype = Variable
    if requires_grad:
        dtype = Parameter
    operator = get_np_operator(name, in_channels, out_channels)
    return dtype(cuda_cast(torch.from_numpy(operator)), requires_grad=requires_grad)


def cuda_cast(param, dtype='float'):
    return getattr(param.cuda(), dtype)()
