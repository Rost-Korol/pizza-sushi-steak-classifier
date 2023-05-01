import torch
import torchvision.models
from torch import nn
from pathlib import Path
from timeit import default_timer as timer
import random
from PIL import Image
import os

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class_names = ['pizza', 'steak', 'sushi']
guessing_phrases = [
    "Hmm.. I think it's a",
    "Maybe it is a",
    "I'm guessing it's a",
    "From what I can see, it looks like a",
    "It seems to be a",
    "Based on the shape/texture/color, it could be a",
    "My best guess would be a",
    "It appears to be a",
    "Judging by the features, it might be a",
    "I'm inclined to say it's a",
    "If I had to guess, I'd say it's a",
    "To me, it looks like a",
    "The characteristics suggest it could be a",
    "Upon closer examination, it resembles a",
    "Let me consult my crystal ball... it's a",
    "My spidey senses are tingling... it's a",
    "I'm getting a strong food vibe... it's a",
    "The force is telling me... it's a",
    "After careful analysis, I've determined it's a",
    "I'm sensing some serious food energy... it's a",
    "I'm a machine learning model, not a magician... but it's a",
    "My AI-powered brain says it's a",
    "My neural network is firing on all cylinders... it's a",
    "According to my calculations, it's a",
    "My image recognition skills are top-notch... it's a",
    "If I were a betting AI, I'd put money on it being a",
    "I'm 99.99% confident it's a",
    "I'm pretty sure it's a",
    "My image classification algorithms are second to none... it's a"
]

# # Write transform for turning images into tensors
# image_transform = transforms.Compose([
#     transforms.Resize(size=(64, 64)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x[:3, :, :])  # Leave only RGB
# ])

image_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

########################################################################################################################
# 2 conv layers + 1 linear layers
class TinyVGG_1(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int, ):
        super().__init__()
        # conv layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # conv layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # classifier layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13,
                      out_features=output_shape)
        )
    def forward(self, x):
        return self.classifier(self.conv2(self.conv1(x)))


# def load_model():
#     model = TinyVGG_1(input_shape=3, hidden_units=20, output_shape=3)
#     model.load_state_dict(torch.load(f='model/food_classificator_7883.pth'))
#     return model

def load_model():
    model = torchvision.models.efficientnet_b2()
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=3, bias=True)
    )
    model.load_state_dict(torch.load(f='model/07_effb2_100_percent_20_epochs.pth'))
    return model




