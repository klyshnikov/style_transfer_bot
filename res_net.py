# ............Imports............
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import json
import numpy as np

def resnet_train(user_id):
    #.............Читаем переменные...........
    with open('data_file.json', 'r') as j:
        json_data = json.load(j)
    closs_factor = int(json_data[user_id]["closs_factor"])
    sloss_factor = int(json_data[user_id]["sloss_factor"])
    num_steps = int(json_data[user_id]["num_epochs"])
    is_big = int(json_data[user_id]["is_big"])
    imsize = int(json_data[user_id]["im_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #.................Ставим функции для обрезки изображений..............

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])

    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(user_id + "style_sqr.jpg")
    content_img = image_loader(user_id + "content_sqr.jpg")
    unloader = transforms.ToPILImage()
    
    #...............Функции потерь контента.............

    class ContentLoss(nn.Module):
        def __init__(self, target, ):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    
    #...................Функция потери стиля...............

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    #....................Нормализация...................

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std
        
    #.................Функция сборки модели................

    def get_style_model_and_losses(normalization_mean, normalization_std,
                                   style_img, content_img):
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)

        # adding

        model.add_module("conv_1", nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1)))

        target_style_1 = model(style_img).detach()
        style_loss_1 = StyleLoss(target_style_1)
        model.add_module("style_loss_1", style_loss_1)
        style_losses.append(style_loss_1)

        model.add_module("relu_1", nn.ReLU())

        model.add_module("conv_2", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)))

        target_style_2 = model(style_img).detach()
        style_loss_2 = StyleLoss(target_style_2)
        model.add_module("style_loss_2", style_loss_2)
        style_losses.append(style_loss_2)

        model.add_module("relu_2", nn.ReLU())

        model.add_module("max_pool_2", nn.MaxPool2d(kernel_size=2))

        model.add_module("conv_3", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)))

        target_style_3 = model(style_img).detach()
        style_loss_3 = StyleLoss(target_style_3)
        model.add_module("style_loss_3", style_loss_3)
        style_losses.append(style_loss_3)

        model.add_module("relu_3", nn.ReLU())

        model.add_module("conv_4", nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1)))

        target_content_4 = model(content_img).detach()
        content_loss_4 = ContentLoss(target_content_4)
        model.add_module("content_loss_4", content_loss_4)
        content_losses.append(content_loss_4)

        model.add_module("relu_4", nn.ReLU())

        model.add_module("upsample_4", nn.Upsample(scale_factor=2))

        model.add_module("conv_5", nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=(1, 1)))

        target_style_5 = model(style_img).detach()
        style_loss_5 = StyleLoss(target_style_5)
        model.add_module("style_loss_5", style_loss_5)
        style_losses.append(style_loss_5)

        model.add_module("relu_5", nn.ReLU())

        model.add_module("conv_6", nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), padding=(1, 1)))

        target_style_6 = model(style_img).detach()
        style_loss_6 = StyleLoss(target_style_6)
        model.add_module("style_loss_6", style_loss_6)
        style_losses.append(style_loss_6)

        target_content_6 = model(content_img).detach()
        content_loss_6 = ContentLoss(target_content_6)
        model.add_module("content_loss_6", content_loss_6)
        content_losses.append(content_loss_6)

        return model, style_losses, content_losses

    input_img = content_img.clone()

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer
    
    #.......................Основная функция - тренировка..................

    def run_style_transfer(normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=num_steps,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(normalization_mean, normalization_std,
                                                                         style_img,
                                                                         content_img)
        optimizer = get_input_optimizer(input_img)

        print(model)
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = sloss_factor * style_score + closs_factor * content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)
        print(model(style_img).shape)

        return input_img

    output = run_style_transfer(cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    
    #......................Кривое сохранение изображения.................

    k = np.array(output[0].detach().numpy())
    k_reshape = np.einsum('kij->ijk', k)
    print(k_reshape.shape)
    plt.imsave((user_id + 'output.jpg'), k_reshape)
