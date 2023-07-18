# -*- coding: utf-8 -*-
# @Time : 2023/4/10 14:39
# @Author : zihua.zeng
# @File : test.py


import os

import cv2
import torch
from torchvision import transforms
from timm.data import create_dataset, create_loader
from timm.models import create_model


def test_ds():
    data_path = "/Users/zihua.zeng/Dataset/uneven_0527_ds"
    dataset = create_dataset(
        root=data_path,
        name="",
        split="training",
        class_map=None,
    )

    loader = create_loader(
        dataset,
        input_size=(3, 320, 320),
        batch_size=1,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=(0.48145, 0.45782, 0.40821073),
        std=(0.26862, 0.2613, 0.2757),
        num_workers=4,
        color_jitter=0.4,
        vflip=0.5,
        crop_pct=1.0,
        crop_mode="center",
        pin_memory=False,
        device="cpu",
        tf_preprocessing=False,
    )

    dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(320),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4814, 0.4578, 0.4082], [0.2686, 0.2613, 0.2757])
    ])

    for i, (img, target) in enumerate(loader):
        print(img.shape)
        print(target)
        break

    from IPython import embed
    embed()


def test_model():
    model_name = "convnext_base.clip_laion2b"
    # model_name = "resnet50"
    num_classes = 2
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=3,
        global_pool=None,
        scriptable=False
    )
    # for param in model.stem.parameters():
    #     param.requires_grad = False
    # for param in model.stages.parameters():
    #     param.requires_grad = False
    # for param in model.norm_pre.parameters():
    #     param.requires_grad = False

    from torchprofile import profile_macs
    img = torch.randn(1, 3, 320, 320)
    macs = profile_macs(model, img)
    print(macs / 1e9)


def create_split_screen_dataset():
    import shutil
    import random
    from glob import glob
    import numpy as np
    from pathlib import Path

    image_path = "/home/zihua/data_warehouse/datasuit_db/abnormal_sampler_0404"
    output_path = "/home/zihua/data_warehouse/split_image"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(output_path, "Positive").mkdir(parents=True, exist_ok=True)
    Path(output_path, "Negative").mkdir(parents=True, exist_ok=True)
    fns = glob(os.path.join(image_path, "*.jpg"))

    negatives = random.sample(fns, 10000)

    for fn in negatives:
        shutil.copy(fn, os.path.join(output_path, "Negative", os.path.basename(fn)))

    split_choices = [2, 4]
    cat_choices = ["hor", "ver"]
    pos_num = 4000
    for _ in range(pos_num):
        choice = random.choice(split_choices)
        if choice == 2:
            img1 = None
            img2 = None
            while img1 is None or img2 is None:
                img1 = cv2.imread(random.choice(fns))
                img2 = cv2.imread(random.choice(fns))

            out_size = random.choice([img1.shape[0], img2.shape[0]])

            cat = random.choice(cat_choices)
            if cat == "hor":
                left_start = random.randint(0, (img1.shape[1]) // 2)
                left = img1[:, left_start:(left_start + img1.shape[1] // 2), :]
                left = cv2.resize(left, (out_size // 2, out_size))
                right_start = random.randint(0, (img2.shape[1]) // 2)
                right = img2[:, right_start:(right_start + img2.shape[1] // 2), :]
                right = cv2.resize(right, (out_size // 2, out_size))
                out = np.hstack([left, right])
                cv2.imwrite(os.path.join(output_path, "Positive", f"{_}.jpg"), out)
            else:
                top_start = random.randint(0, (img1.shape[0]) // 2)
                top = img1[top_start:(top_start + img1.shape[0] // 2), :, :]
                top = cv2.resize(top, (out_size, out_size // 2))
                bottom_start = random.randint(0, (img2.shape[0]) // 2)
                bottom = img2[bottom_start:(bottom_start + img2.shape[0] // 2), :, :]
                bottom = cv2.resize(bottom, (out_size, out_size // 2))
                out = np.vstack([top, bottom])
                cv2.imwrite(os.path.join(output_path, "Positive", f"{_}.jpg"), out)
        else:
            img1, img2, img3, img4 = None, None, None, None
            while img1 is None or img2 is None or img3 is None or img4 is None:
                img1 = cv2.imread(random.choice(fns))
                img2 = cv2.imread(random.choice(fns))
                img3 = cv2.imread(random.choice(fns))
                img4 = cv2.imread(random.choice(fns))

            out_size = random.choice([img1.shape[0], img2.shape[0], img3.shape[0], img4.shape[0]])
            remainder = out_size % 4
            out_size += remainder

            imgs = [img1, img2, img3, img4]
            random.shuffle(imgs)
            top_left = cv2.resize(imgs[0], (out_size // 4, out_size // 4))
            top_right = cv2.resize(imgs[1], (out_size // 4, out_size // 4))
            top_part = np.hstack([top_left, top_right])
            bottom_left = cv2.resize(imgs[2], (out_size // 4, out_size // 4))
            bottom_right = cv2.resize(imgs[3], (out_size // 4, out_size // 4))
            bottom_part = np.hstack([bottom_left, bottom_right])

            out = np.vstack([top_part, bottom_part])
            cv2.imwrite(os.path.join(output_path, "Positive", f"{_}.jpg"), out)


def clear_useless_image():
    import os
    from tqdm import tqdm
    from glob import glob
    image_path = "/home/zihua/data_warehouse/datasuit_db/abnormal_sampler_0404"
    for n in tqdm(glob(os.path.join(image_path, "*.jpg"))):
        img = cv2.imread(n)
        if img is None:
            os.remove(n)


if __name__ == '__main__':
    # test_model()
    # test_ds()
    create_split_screen_dataset()
    # clear_useless_image()
    pass
