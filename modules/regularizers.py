# -*- coding: utf-8 -*-
# Aydın Göze Polat, 2017
# Image regularizers as PyTorch modules

import torch
import torch.nn.functional as F
from torch import nn
from utils import scalar_to_parameter, make_operator


class PeronaMalik(nn.Module):
    def get_exp_diffusion_coefficient(self):
        return torch.exp(-torch.pow(self.gradients, 2) / self.diffusion_rate_2)

    def get_inv_diffusion_coefficient(self):
        return torch.pow(self.gradient_magnitude / self.diffusion_rate_2 + 1, -1)

    def pick_coefficient(self, coefficient):
        # pick the method which will calculate the diffusion speed
        if coefficient == 'exp':
            self.coefficient = self.get_exp_diffusion_coefficient
        elif coefficient == "inv":
            self.coefficient = self.get_inv_diffusion_coefficient

    def get_conv_diffusion(self, image, padded_image):
        # experimental, doesn't work well
        return image * F.tanh(F.conv2d(1-padded_image, self.conv1_params))

    def init_weights(self, layer):
        nn.init.orthogonal(layer.weight)
        nn.init.constant(layer.bias, 0.0)

    def __init__(self, size, diffusion_rate, delta_t, coefficient='exp', learn_operator=False):
        super(PeronaMalik, self).__init__()
        self.gradients = None
        self.delta_t = delta_t
        self.gradient_magnitude = None

        # allow learning parameters using requires_grad=True
        self.diffusion_rate_2 = scalar_to_parameter(diffusion_rate ** 2, size, requires_grad=False)
        self.conv1_params = make_operator('laplace', requires_grad=True)

        self.laplace = make_operator('laplace', requires_grad=learn_operator)
        self.scharr_x = make_operator('scharr_x')
        self.scharr_y = make_operator('scharr_y')

        self.coefficient = None
        self.pick_coefficient(coefficient)

    def forward(self, image):
        # ReflectionPad2d for the image boundaries instead of padding=1
        padded_image = nn.ReflectionPad2d(1)(image)
        self.gradients = F.conv2d(padded_image, self.laplace)

        if self.coefficient == self.get_inv_diffusion_coefficient:
            grad_x = F.conv2d(padded_image, self.scharr_x)
            grad_y = F.conv2d(padded_image, self.scharr_y)
            self.gradient_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))

        if self.coefficient is None:
            diffusion = self.gradients * torch.pow(
                 torch.abs(self.get_conv_diffusion(image, padded_image)) / self.diffusion_rate_2 + 1, -1)
        else:
            diffusion = self.gradients * self.coefficient()
        image = image + diffusion * self.delta_t

        return torch.clamp(image, min=0, max=1.0)