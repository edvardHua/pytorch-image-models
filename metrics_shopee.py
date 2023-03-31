# -*- coding: utf-8 -*-
# @Time : 2023/3/22 11:54
# @Author : zihua.zeng
# @File : metrics_shopee.py
import os
import time
import json

import cv2
import requests
from pprint import pprint
import numpy as np


def calc_prcision_recall(gt, pred, cls_id):
    """

    :param gt: list of gt classes index
    :param pred: list of pred classes index
    :param cls_id:
    :return:
    """

    true_positive = 0
    false_negative = 0
    all_positive = 0
    for g, p in zip(gt, pred):
        if (g == p) and g == cls_id:
            true_positive += 1

        if g == cls_id:
            all_positive += 1

        if p == cls_id and g != p:
            false_negative += 1
    print(true_positive, false_negative, all_positive)
    if true_positive == 0 and false_negative == 0:
        return 0, 0, 0

    prec = (true_positive / (true_positive + false_negative))
    recall = true_positive / all_positive
    if prec == 0 and recall == 0:
        f1score = 0
    else:
        f1score = 2 * prec * recall / (prec + recall)

    return prec, recall, f1score


def test1():
    npzfile = np.load("pred_gt.npz")

    threshold = 0.11761720478534698

    pred = npzfile['arr_0'].tolist()
    gt = npzfile['arr_1'].tolist()
    pred_filter = []
    postive_value = []
    for ind, val in enumerate(pred):
        if val[1] >= threshold:
            pred_filter.append(1)
        else:
            pred_filter.append(0)

        if gt[ind] == 1:
            postive_value.append(val[1])

    m = calc_prcision_recall(gt, pred_filter, 1)
    pprint(m)
    pprint(list(sorted(postive_value)))


def test2():
    left_path = "/Users/zihua.zeng/Dataset/暗亮分类/train_ds_0322/validation/normal"
    right_path = "/Users/zihua.zeng/Dataset/暗亮分类/train_ds_0322/training/normal"

    right_db = {}
    left_db = {}

    for fn in os.listdir(right_path):
        if not fn.endswith(".jpg"):
            continue
        right_db[fn] = 1

    hit_count = 0
    for fn in os.listdir(left_path):
        if not fn.endswith(".jpg"):
            continue
        if fn in right_db:
            hit_count += 1

    print(hit_count)


def test3():
    anno_json_path = "/Users/zihua.zeng/Workspace/LowQualityClassifier/data/al-underexposure-0310-part4_under_0310_chunk0_output_20230322_145137.json"
    # out_dir_path = "/Users/zihua.zeng/Dataset/暗亮分类/test_ds_0322"
    # os.makedirs(out_dir_path, exist_ok=True)
    # cls_names = ['Normal', 'Underexposure']
    # for cn in cls_names: os.makedirs(os.path.join(out_dir_path, cn), exist_ok=True)

    right_path = "/Users/zihua.zeng/Dataset/暗亮分类/train_ds_0322/training/normal"
    right_db = {}
    for fn in os.listdir(right_path):
        if not fn.endswith(".jpg"):
            continue
        right_db[fn] = 1

    normal_count = 0
    underexposure_count = 0
    hit_count = 0
    not_sure_count = 0
    conflict_count = 0
    anno = json.load(open(anno_json_path))
    for ind, item in enumerate(anno):

        if len(item['labels']) < 2:
            continue

        res1 = item['labels'][0]['result'][0]['value']['choices'][0]
        res2 = item['labels'][1]['result'][0]['value']['choices'][0]

        if res1 != res2:
            conflict_count += 1
            # img_data = requests.get(item['data']['image']).content
            # img_fn = os.path.basename(item['data']['image'])
            # f = open(os.path.join(out_dir_path, cls_names[3], '%s.jpg' % (img_fn)), "wb")
            # f.write(img_data)
            # f.close()
            # time.sleep(0.2)
            continue

        if res1 == "Normal":
            img_data = requests.get(item['data']['image']).content
            img_fn = os.path.basename(item['data']['image'])

            if "%s.jpg" % img_fn in right_db:
                hit_count += 1

            with open(os.path.join("/Users/zihua.zeng/Dataset/暗亮分类/train_ds_0322/training/normal", img_fn + ".jpg"),
                      "wb") as handler:
                handler.write(img_data)

            normal_count += 1
            if normal_count == 2500:
                break

            time.sleep(0.1)
        elif res1 == "Underexposure":
            underexposure_count += 1

    print(hit_count)


def test4():
    base_path = "/Users/zihua.zeng/Dataset/暗亮分类/train_ds_0322/validation/normal"
    c = 0
    for ip in os.listdir(base_path):
        if not ip.endswith(".jpg"):
            continue
        image = cv2.imread(os.path.join(base_path, ip))

        if image is None:
            os.remove(os.path.join(base_path, ip))
            c += 1
    print(c)


if __name__ == '__main__':
    test4()
    pass
