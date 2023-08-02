# -*- coding: utf-8 -*-
# @Time : 2023/4/3 16:42
# @Author : zihua.zeng
# @File : inference.py

import os
import sys
import random
import torch
import shutil
import numpy as np

from tqdm import tqdm
from timm.data import create_dataset, create_loader
from timm.models import create_model, load_checkpoint
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecallAtFixedPrecision


def model_metric(data_path, checkpoint, inpsize, model_name="convnext_base.clip_laion2b"):
    """
    对模型进行评估，且是固定不同的 fixed precision 下的 PR 值

    """
    device = "cuda"
    # model_name = "convnext_base.clip_laion2b"
    num_classes = 2
    # data_path = "/home/zihua/data_warehouse/uneven_0511/validation"
    # checkpoint = "output/train/uneven_0510/model_best.pth.tar"

    dataset = create_dataset(
        root=data_path,
        name="",
        split="",
        class_map=None
    )

    loader = create_loader(
        dataset,
        is_training=False,
        input_size=(3, eval(inpsize), eval(inpsize)),
        batch_size=32,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=(0.48145, 0.45782, 0.40821073),
        std=(0.26862, 0.2613, 0.2757),
        num_workers=4,
        crop_pct=1.0,
        crop_mode="center",
        pin_memory=False,
        # device="cpu",
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
        model.to("cuda")

    output_batches = []
    target_batches = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if device == "cuda":
                data = data.to("cuda")
            output = model(data)
            output = torch.nn.functional.softmax(output, dim=-1)
            output_batches.append(output)
            target_batches.append(target)

    output_all = torch.cat(output_batches, dim=0)
    target_all = torch.cat(target_batches, dim=0)

    output_all = output_all.detach().cpu()
    target_all = target_all.detach().cpu()

    precssss = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.8, 0.9]
    prec_arr = []
    recall_arr = []
    threshold_arr = []
    for prec in precssss:
        mcrafp = MulticlassRecallAtFixedPrecision(num_classes=num_classes, min_precision=prec, thresholds=None)

        recalls, thresholds = mcrafp(output_all, target_all)

        # print("mcr recall is:", recalls[-1])
        # print("mcr threshold is:", thresholds[-1])

        ##pred_probs = torch.nn.functional.softmax(output_all, dim=-1)
        pred_probs = output_all
        pred_all = (pred_probs[:, -1] >= thresholds[-1]).int()
        ##print("pred_all shape:", pred_all.shape)

        # mcf1s = MulticlassF1Score(num_classes=args.num_classes, average=None).to(device)
        confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        mcr = MulticlassRecall(num_classes=num_classes, average=None)
        mcp = MulticlassPrecision(num_classes=num_classes, average=None)

        precision = mcp(pred_all, target_all)[-1]
        recall = mcr(pred_all, target_all)[-1]

        prec_arr.append(precision)
        recall_arr.append(recall)
        threshold_arr.append(thresholds[-1])

        # print("precision is {}, recall is {}, threshold is {}".format(precision, recall, thresholds[-1]))

        # f1 = mcf1s(output_all, target_all)[-1]
        cmetrics = confmat(pred_all, target_all)
        print("conf metrix: %.2f" % prec, cmetrics)

    print(dataset.reader.class_to_idx)
    print("Fixed Precision, Precision, Recall, Threshold")
    for i, j, k, l in zip(precssss, prec_arr, recall_arr, threshold_arr):
        print("%.4f, %.4f, %.4f, %.6f" % (i, j, k, l))
    
    from IPython import embed
    embed()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        model_metric(sys.argv[1], sys.argv[2], sys.argv[3])
    if len(sys.argv) == 5:
        model_metric(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        
