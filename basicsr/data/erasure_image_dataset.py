import random
from os import path as osp

import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import crop, normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ErasureImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(ErasureImageDataset, self).__init__()
        self.opt = opt
        self.decode = opt.get("decode", True)
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.lq_folder = opt["dataroot_lq"]
        self.gt_folder = opt["dataroot_gt"]

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [
                self.lq_folder,
                self.lq_folder,
                self.gt_folder,
            ]
            self.io_backend_opt["client_keys"] = ["lq1", "lq2", "identity"]
            self.lq_paths1 = paths_from_lmdb(self.lq_folder)
            self.lq_paths2 = paths_from_lmdb(self.lq_folder)
            self.identity_paths = paths_from_lmdb(self.gt_folder) + paths_from_lmdb(
                self.lq_folder
            )
        elif "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.lq_paths1 = [
                    osp.join(self.lq_folder, line.rstrip().split(" ")[0])
                    for line in fin
                ]
                self.lq_paths2 = [
                    osp.join(self.lq_folder, line.rstrip().split(" ")[0])
                    for line in fin
                ]
                self.identity_paths = [
                    osp.join(self.gt_folder, line.rstrip().split(" ")[0])
                    for line in fin
                ] + self.lq_paths1
        else:
            self.lq_paths1 = sorted(list(scandir(self.lq_folder, full_path=True)))
            self.lq_paths2 = sorted(list(scandir(self.lq_folder, full_path=True)))
            self.identity_paths = (
                sorted(list(scandir(self.gt_folder, full_path=True))) + self.lq_paths1
            )

        random.shuffle(self.lq_paths2)
        for _ in range(2):
            random.shuffle(self.identity_paths)

        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        lq1, lq1_path = self.get(index, self.lq_paths1, "lq1")
        lq2, lq2_path = self.get(index, self.lq_paths2, "lq2")
        identity1, identity1_path = self.get(index, self.identity_paths, "identity")
        identity2, identity2_path = self.get(index, self.identity_paths, "identity")
        identity = torch.stack([identity1, identity2])

        return {
            "lq1": lq1,
            "lq2": lq2,
            "lq1_path": lq1_path,
            "lq2_path": lq2_path,
            "identity": identity,
            "identity1_path": identity1_path,
            "identity2_path": identity2_path,
        }

    def __len__(self):
        return len(self.lq_paths1)

    def get(self, index, paths, key="lq1"):
        # load lq image
        img_path = paths[index]
        img_bytes = self.file_client.get(img_path, key)
        if self.decode:
            img = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img = img[3:].reshape(h, w, c)
            if self.float32:
                img = img.astype(np.float32) / 255.0

        # flip, rotation
        img = augment([img])

        # BGR to RGB
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC to CHW, numpy to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous()

        # random crop
        c, h, w = img.shape
        img_size = self.opt["img_size"]
        img = crop(
            img,
            top=random.randint(0, h - img_size),
            left=random.randint(0, w - img_size),
            height=img_size,
            width=img_size,
        )

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img, self.mean, self.std, inplace=True)

        return img, img_path
