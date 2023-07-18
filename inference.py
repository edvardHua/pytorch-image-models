# -*- coding: utf-8 -*-
# @Time : 2023/4/3 16:42
# @Author : zihua.zeng
# @File : inference.py

import os
import random
import torch
import shutil
import numpy as np

from timm.data import create_dataset, create_loader
from timm.models import create_model, load_checkpoint


def infer_folder():
    """
    对文件夹的图片进行一张张的推断，只适配 batch_size = 1 的情况
    TODO: 这里的 inference 有点问题
    """
    # model_name = "convnext_base.clip_laion2b"
    model_name = "resnet50"
    device = "cuda"
    inpsize = 368
    num_classes = 2
    data_path = "/home/zihua/data_warehouse/datasuit_db/abnormal_sampler_0404"
    # 用于区别 uneven 和 normal
    checkpoint = "/home/zihua/data_warehouse/low_quality/ablight_model/resnet50_split_image/model_best.pth.tar"
    idx_threshold = {1: 0.3}
    out_path_normal = None
    out_path_uneven = "/home/zihua/data_warehouse/low_quality/infer_split"

    if out_path_normal is not None:
        os.makedirs(out_path_normal, exist_ok=True)
    os.makedirs(out_path_uneven, exist_ok=True)

    dataset = create_dataset(
        root=data_path,
        name="",
        split="",
        class_map=None
    )

    loader = create_loader(
        dataset,
        is_training=False,
        input_size=(3, inpsize, inpsize),
        batch_size=1,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=(0.48145, 0.45782, 0.40821073),
        std=(0.26862, 0.2613, 0.2757),
        num_workers=4,
        crop_pct=1.0,
        crop_mode="center",
        pin_memory=False,
        device=device,
        tf_preprocessing=False,
    )

    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=3,
        global_pool=None,
        scriptable=False
    )

    load_checkpoint(model, checkpoint, False)
    model.eval()

    if device == "cuda":
        model = model.to(device)

    counter = 0
    output_batches = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if device == "cuda":
                data = data.to("cuda")
            output = model(data)
            output = torch.nn.functional.softmax(output, dim=-1)
            output_batches.append(output)
            ip = dataset.filename(batch_idx)
            for k, v in idx_threshold.items():
                if output[0][k] >= v:
                    shutil.copy(os.path.join(data_path, ip), os.path.join(out_path_uneven, ip))
                    counter += 1
                else:
                    if out_path_normal is not None:
                        shutil.copy(os.path.join(data_path, ip), os.path.join(out_path_normal, ip))
            print("%d/%d" % (batch_idx, len(loader)))

        print(counter)


if __name__ == '__main__':
    infer_folder()
    pass
