import os
import json
import random
from enum import Enum

import PIL
import torch
from torchvision import transforms

from config import sub_datasets


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    """
    Different dataset split for different task.
    """
    TRAIN = "train" # train split
    TEST = "test" # test split
    TEST_NG = "test_ng" # test split, but in the semi-supervise setting


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset.
    """

    def __init__(
        self,
        source,
        classname,
        test_filelist_dir,
        resize=256,
        imagesize=256,
        split=DatasetSplit.TRAIN,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the data folder.
            classname: [str or None]. Name of datasets' class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. Size the loaded image initially gets resized to.
            imagesize: [int]. Size the resized loaded image gets (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else sub_datasets
        self.test_filelist_dir = test_filelist_dir


        self.transform_img = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def get_image_data(self):
        imgpaths_per_class = {classname: {} for classname in self.classnames_to_use}
        maskpaths_per_class = {classname: {} for classname in self.classnames_to_use}

        # 读取test filelist
        if self.split is DatasetSplit.TEST_NG:
            with open(self.test_filelist_dir, 'r') as f:
                test_filelist_json = json.load(f)

        for classname in self.classnames_to_use:
            # example: .../mvtec/bottle/train/
            if self.split == DatasetSplit.TRAIN:
                classpath = os.path.join(self.source, classname, "train")
            else:
                classpath = os.path.join(self.source, classname, "test")

            maskpath = os.path.join(self.source, classname, "ground_truth")

            if self.split is DatasetSplit.TRAIN:
                imgpaths_per_class[classname]["good"] = [
                    os.path.join(classpath, "good", x) for x in sorted(os.listdir(os.path.join(classpath, "good")))
                ]
                maskpaths_per_class[classname]["good"] = None

            else:
                # test/ng1, test/ng2, ..., test/good
                anomaly_types = os.listdir(classpath)

                if self.split is DatasetSplit.TEST_NG:
                    subcls_ng_filelist = test_filelist_json[classname]["ng_filelist"]

                for anomaly in anomaly_types:
                    anomaly_path = os.path.join(classpath, anomaly)
                    anomaly_files = sorted(os.listdir(anomaly_path))

                    if self.split is DatasetSplit.TEST_NG:
                        imgpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_path, x) for x in anomaly_files
                            if os.path.join(anomaly, x) not in subcls_ng_filelist
                        ]
                        maskpaths_per_class[classname][anomaly] = (
                            None if anomaly == "good" else [
                                os.path.join(maskpath, anomaly, x) for x in
                                sorted(os.listdir(os.path.join(maskpath, anomaly)))
                                if os.path.join(anomaly, x) not in subcls_ng_filelist
                            ]
                        )

                    else:
                        imgpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_path, x) for x in anomaly_files
                        ]
                        maskpaths_per_class[classname][anomaly] = (
                            None if anomaly == "good" else [
                                os.path.join(maskpath, anomaly, x) for x in
                                sorted(os.listdir(os.path.join(maskpath, anomaly)))
                            ]
                        )

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split is DatasetSplit.TEST and anomaly != "good" \
                            or self.split is DatasetSplit.TEST_NG and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split is DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        elif self.split is DatasetSplit.TEST_NG and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)


class NGDataset(torch.utils.data.Dataset):
    """
    NG Dataset.
    """

    def __init__(
        self,
        source,
        classname,
        test_filelist_dir,
        resize=256,
        imagesize=256,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the data folder.
            classname: [str or None]. Name of datasets' class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. Size the loaded image initially gets resized to.
            imagesize: [int]. Size the resized loaded image gets (center-)cropped to.
        """
        super().__init__()
        self.source = source
        self.classnames_to_use = [classname] if classname is not None else sub_datasets
        self.test_filelist_dir = test_filelist_dir

        self.transform_img = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

        self.data_to_iterate, self.mask_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        image_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask_path = self.mask_to_iterate[idx]
        mask = PIL.Image.open(mask_path)
        mask = self.transform_mask(mask)

        return {
            "image": image,
            "mask": mask,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):

        seen_anomaly_list = []
        seen_anomaly_mask_list = []

        with open(self.test_filelist_dir, 'r') as f:
            test_filelist_json = json.load(f)

        for classname in self.classnames_to_use:
            subcls_ng_filelist = test_filelist_json[classname]["ng_filelist"]
            sub_seen_anomaly_list = [os.path.join(self.source, classname, "test", x)
                                     for x in subcls_ng_filelist]
            seen_anomaly_list.extend(sub_seen_anomaly_list)

            sub_seen_anomaly_mask_list = [os.path.join(self.source, classname, "ground_truth", x.replace(".jpg", ".png"))
                                          for x in subcls_ng_filelist]
            seen_anomaly_mask_list.extend(sub_seen_anomaly_mask_list)

        return seen_anomaly_list, seen_anomaly_mask_list
