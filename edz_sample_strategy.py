# -*- coding: utf-8 -*-
# @Time : 2023/5/18 11:29
# @Author : zihua.zeng
# @File : dataset_edz.py


import os
import shutil
import datetime

from glob import glob


class EDZDSSample(object):

    def __init__(self, data_path, eval_ratio=0.2,
                 out_suffix=None, cls_betw_ratio=None):
        """
        :param data_path: 数据集路径
        :param eval_ratio: 验证集比例
        :param cls_betw_ratio: 类别不平衡比例, list, [3, 1]
        """
        self.data_path = data_path
        self.eval_ratio = eval_ratio
        self.cls_betw_ratio = cls_betw_ratio
        self.out_suffix = out_suffix
        self.cls_list = list(filter(lambda _: os.path.isdir(_), glob("%s/*" % self.data_path)))

    def sample_ds(self, data_path, out_path, val_ratio=0.2):
        """
        随机采样 data_path，生成对应的 train 和 val
        :param data_path:
        :param out_path:
        :return:
        """
        import shutil
        from pathlib import Path

        train_path = os.path.join(out_path, "training")
        val_path = os.path.join(out_path, "validation")
        Path(train_path).mkdir(parents=True, exist_ok=True)
        Path(val_path).mkdir(parents=True, exist_ok=True)

        for dn in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, dn)) is False:
                continue
            tmp = os.listdir(os.path.join(data_path, dn))
            tmp = list(filter(lambda _: _.endswith(".jpg"), tmp))
            split_nums = int(len(tmp) * (1 - val_ratio))
            tfs = tmp[:split_nums]
            vfs = tmp[split_nums:]

            Path(os.path.join(train_path, dn)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(val_path, dn)).mkdir(parents=True, exist_ok=True)
            for v in tfs: shutil.copy(os.path.join(data_path, dn, v),
                                      os.path.join(train_path, dn, v))
            for v in vfs: shutil.copy(os.path.join(data_path, dn, v),
                                      os.path.join(val_path, dn, v))

    def __call__(self):
        out_dir_path = self.data_path.replace(
            os.path.basename(self.data_path),
            "edz_sample_ds"
        )
        os.makedirs(out_dir_path, exist_ok=True)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y_%m_%d")

        cls_ratio_str = ""
        if self.cls_betw_ratio is None:
            cls_ratio_str = "avg"
        else:
            # 把 cls_betw_ratio 转成 str
            cls_ratio_str = "cls_betw_" + "_".join(str(_) for _ in self.cls_betw_ratio)

        if self.out_suffix is None:
            self.out_suffix = ""

        out_dir_path = os.path.join(out_dir_path, "%s_%s_%s" % (now_str, cls_ratio_str, self.out_suffix))
        os.makedirs(out_dir_path, exist_ok=True)
        # 先按照类别比例采样
        self.sample_ds(self.data_path,
                       out_dir_path,
                       self.eval_ratio)
        # 再按类间采样
        if self.cls_betw_ratio is not None:
            min_ind = self.cls_betw_ratio.index(min(self.cls_betw_ratio))
            min_sums_train = len(os.listdir(os.path.join(out_dir_path, "training", self.cls_list[min_ind])))
            min_sums_eval = len(os.listdir(os.path.join(out_dir_path, "validation", self.cls_list[min_ind])))

            for idx, cl in enumerate(self.cls_list):
                if idx == min_ind:
                    continue
                # 调整 train
                desire_cls_nums = min_sums_train * self.cls_betw_ratio[idx]
                cur_cls_fp = os.listdir(os.path.join(out_dir_path, "training", cl))
                for _ in range(len(cur_cls_fp) - desire_cls_nums):
                    suffix = cur_cls_fp[_].split(".")[-1]
                    if suffix.lower() not in ["jpg", "png"]:
                        continue
                    shutil.rmtree(os.path.join(out_dir_path, "training", cl, cur_cls_fp[_]))
                # 调整 eval
                desire_cls_nums = min_sums_eval * self.cls_betw_ratio[idx]
                cur_cls_fp = os.listdir(os.path.join(out_dir_path, "validation", cl))
                for _ in range(len(cur_cls_fp) - desire_cls_nums):
                    suffix = cur_cls_fp[_].split(".")[-1]
                    if suffix.lower() not in ["jpg", "png"]:
                        continue
                    shutil.rmtree(os.path.join(out_dir_path, "validation", cl, cur_cls_fp[_]))

        return out_dir_path


if __name__ == '__main__':
    sam = EDZDSSample("/Users/zihua.zeng/Dataset/uneven_dist",
                      cls_betw_ratio=[3, 1])
    path = sam()
    print(path)
    pass
