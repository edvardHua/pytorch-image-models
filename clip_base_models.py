# -*- coding: utf-8 -*-
# @Time : 2023/4/11 10:29
# @Author : zihua.zeng
# @File : clip_base_models.py

import torch
import clip
import torch.nn as nn


class ClipClassifier(nn.Module):

    def __init__(self, num_classes, fixed_feature_extractor=False):
        super(ClipClassifier, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("RN50", device=device)
        self.model = model
        if fixed_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model.encode_image(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 224, 224)
    model = ClipClassifier(2)
    output = model(dummy_input)
    pass
