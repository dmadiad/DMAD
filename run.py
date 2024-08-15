import logging
import json
import os
import sys
import time
import numpy as np
import torch
import random
from itertools import cycle

from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(dataset_name)

import backbones
import common
import metrics
import modle
import my_utils

from datasets.base import DatasetSplit, BaseDataset, NGDataset

device = my_utils.set_torch_device(gpu_id)


def get_model():
    global layers_to_extract_from
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    models = []
    for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
    ):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                backbone_name.split("-")[-1]
            )
        backbone = backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        DMAD_inst = model.DMAD(device)

        DMAD_inst.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=(3, image_size, image_size),
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            patchsize=patchsize,
            embedding_size=embedding_size,
            meta_epochs=meta_epochs,
            aed_meta_epochs=aed_meta_epochs,
            gan_epochs=gan_epochs,
            noise_std=noise_std,
            dsc_layers=dsc_layers,
            dsc_hidden=dsc_hidden,
            dsc_margin=dsc_margin,
            dsc_lr=dsc_lr,
            auto_noise=auto_noise,
            train_backbone=train_backbone,
            cos_lr=cos_lr,
            pre_proj=pre_proj,
            use_IN=0,
            proj_layer_type=proj_layer_type,
            mix_noise=mix_noise,
        )
        models.append(DMAD_inst)

    return models


def create_test_filelist(save_path):
    test_filelist = {}

    for sub_dataset in sub_datasets:
        sub_ng_list = {}
        cur_sub_dataset_test_NGfile_list = []

        sub_NG_classes = [x for x in os.listdir(os.path.join(dataset_path, sub_dataset, "test")) if x != "good"]

        for sub_NG_class in sub_NG_classes:
            cur_sub_NG_files = os.listdir(os.path.join(dataset_path, sub_dataset, "test", sub_NG_class))
            cur_sub_NG_files = [os.path.join(sub_NG_class, x) for x in cur_sub_NG_files]
            cur_sub_dataset_test_NGfile_list.extend(cur_sub_NG_files)

        cur_sub_dataset_test_NGfile_list.sort()

        if mode == "train" or mode == "test" and ng_nums == 0:
            sub_ng_list["ng_filelist"] = None
            sub_ng_list["remain_testfilelist"] = cur_sub_dataset_test_NGfile_list
        else:
            random.seed(seed)
            random.shuffle(cur_sub_dataset_test_NGfile_list)

            sub_ng_list["ng_filelist"] = cur_sub_dataset_test_NGfile_list[:ng_nums]
            sub_ng_list["ng_filelist"].sort()
            sub_ng_list["remain_testfilelist"] = cur_sub_dataset_test_NGfile_list[ng_nums:]
            sub_ng_list["remain_testfilelist"].sort()

        test_filelist[sub_dataset] = sub_ng_list

    # 保存NG list
    json_test_filelist = json.dumps(test_filelist)

    with open(os.path.join(save_path, "test_filelist.json"), 'w') as f:
        f.write(json_test_filelist)

    return os.path.join(save_path, "test_filelist.json")


def get_unify_dataloader(test_filelist_dir):
    test_dataloaders = []

    for i, sub_dataset in enumerate(sub_datasets, start=1):

        if ng_nums == 0:
            test_dataset = BaseDataset(
                dataset_path,
                sub_dataset,
                test_filelist_dir,
                resize=image_size,
                imagesize=image_size,
                split=DatasetSplit.TEST
            )
        else:
            test_dataset = BaseDataset(
                dataset_path,
                sub_dataset,
                test_filelist_dir,
                resize=image_size,
                imagesize=image_size,
                split=DatasetSplit.TEST_NG
            )

        LOGGER.info("Dataset {} {}: test={}".format(i, sub_dataset, len(test_dataset)))

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

        test_dataloaders.append(test_dataloader)

    if mode == "test":
        return test_dataloaders

    elif mode == "train":
        train_dataset = BaseDataset(
            dataset_path,
            None,
            test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
            split=DatasetSplit.TRAIN
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

        LOGGER.info("Dataset: train={}".format(len(train_dataset)))

        return train_dataloader, test_dataloaders

    elif mode == "train_ng":
        train_dataset = BaseDataset(
            dataset_path,
            None,
            test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
            split=DatasetSplit.TRAIN
        )

        NG_dataset = NGDataset(
            dataset_path,
            None,
            test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

        NG_dataloader = torch.utils.data.DataLoader(
            NG_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        LOGGER.info("Dataset: train={} NG={}".format(len(train_dataset), len(NG_dataset)))

        return train_dataloader, test_dataloaders, NG_dataloader


def get_sub_dataloader(subdataset, test_filelist_dir):

    if ng_nums == 0:
        test_dataset = BaseDataset(
            dataset_path,
            classname=subdataset,
            test_filelist_dir=test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
            split=DatasetSplit.TEST
        )
    else:
        test_dataset = BaseDataset(
            dataset_path,
            classname=subdataset,
            test_filelist_dir=test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
            split=DatasetSplit.TEST_NG
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )

    if mode == "test":
        LOGGER.info("Dataset: test={}".format(len(test_dataset)))
        return test_dataloader

    elif mode == "train":
        train_dataset = BaseDataset(
            dataset_path,
            classname=subdataset,
            test_filelist_dir=test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
            split=DatasetSplit.TRAIN
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

        LOGGER.info("Dataset: train={} test={}".format(len(train_dataset), len(test_dataset)))

        return train_dataloader, test_dataloader

    elif mode == "train_ng":
        train_dataset = BaseDataset(
            dataset_path,
            classname=subdataset,
            test_filelist_dir=test_filelist_dir,
            resize=image_size,
            imagesize=image_size,
            split=DatasetSplit.TRAIN
        )

        NG_dataset = NGDataset(
            dataset_path,
            classname=subdataset,
            test_filelist_dir=test_filelist_dir,
            resize=image_size,
            imagesize=image_size
        )


        LOGGER.info("Dataset: train={} test={} NG={}".format(len(train_dataset), len(test_dataset), len(NG_dataset)))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

        NG_dataloader = torch.utils.data.DataLoader(
            NG_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return train_dataloader, test_dataloader, cycle(NG_dataloader)


def run():
    base_save_path = my_utils.create_base_save_folder(dataset_name)

    result_collect = []

    timestamp = int(time.time())
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))

    exp_save_path = my_utils.create_exp_save_folder(base_save_path, time_str)

    log_file_handler = logging.FileHandler(os.path.join(exp_save_path, mode + ".log"))
    LOGGER.addHandler(log_file_handler)

    test_filelist_dir = create_test_filelist(exp_save_path)

    my_utils.fix_seeds(seed, device)

    model_list = get_model()
    models_dir = os.path.join(base_save_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    if setting == "one_class":
        # traverse sub datasets
        for ds_count, sub_dataset in enumerate(sub_datasets, start=1):
            LOGGER.info(
                "Evaluating dataset [{}] ({}/{})...".format(
                    sub_dataset,
                    ds_count,
                    len(sub_datasets),
                )
            )

            for i, DMAD in enumerate(model_list):
                # torch.cuda.empty_cache()
                if DMAD.backbone.seed is not None:
                    my_utils.fix_seeds(DMAD.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(model_list))
                )
                # torch.cuda.empty_cache()

                DMAD.set_model_dir(os.path.join(models_dir, "{}".format(i)), sub_dataset)

                if mode == "train" or mode == "train_ng":
                    if mode == "train":
                        train_dataloader, test_dataloader = get_sub_dataloader(sub_dataset, test_filelist_dir)
                        i_auroc, p_auroc, pro_auroc = DMAD.train(train_dataloader, test_dataloader)
                    else:
                        train_dataloader, test_dataloader, NG_dataloader = get_sub_dataloader(sub_dataset, test_filelist_dir)
                        i_auroc, p_auroc, pro_auroc = DMAD.train(train_dataloader, test_dataloader, NG_dataloader)

                    result_collect.append(
                        {
                            "dataset_name": sub_dataset,
                            "instance_auroc": i_auroc,  # auroc,
                            "full_pixel_auroc": p_auroc,  # full_pixel_auroc,
                            "anomaly_pixel_auroc": pro_auroc,
                        }
                    )

                    for key, item in result_collect[-1].items():
                        if key != "dataset_name":
                            LOGGER.info("{0}: {1:3.3f}".format(key, item))
                            # Store all results and mean scores to a csv-file.
                            result_metric_names = list(result_collect[-1].keys())[1:]
                            result_dataset_names = [results["dataset_name"] for results in result_collect]
                            result_scores = [list(results.values())[1:] for results in result_collect]
                            my_utils.compute_and_store_final_results(
                                exp_save_path,
                                result_scores,
                                column_names=result_metric_names,
                                row_names=result_dataset_names,
                            )

                elif mode == "test":
                    test_dataloader = get_sub_dataloader(sub_dataset, test_filelist_dir)
                    DMAD.gen_anomap(test_dataloader, dataset_name, dataset_path,
                                         sub_dataset, time_str)

            LOGGER.info("\n\n-----\n")

    else:
        # unify setting
        LOGGER.info("Evaluating dataset [{}]:".format(dataset_name))

        for i, DMAD in enumerate(model_list):
            # torch.cuda.empty_cache()
            if DMAD.backbone.seed is not None:
                my_utils.fix_seeds(DMAD.backbone.seed, device)
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(model_list))
            )

            DMAD.set_model_dir(os.path.join(models_dir, "{}".format(i)), '')

            if mode == "train" or mode == "train_ng":
                if mode == "train":
                    train_dataloader, test_dataloaders = get_unify_dataloader(test_filelist_dir)
                    # TODO
                    # prepare core set
                    DMAD.create_coreset(train_dataloader)
                    i_auroc, p_auroc, pro_auroc = DMAD.train(train_dataloader, test_dataloaders,
                                                                  setting="unify")
                else:
                    train_dataloader, test_dataloaders, NG_dataloader = get_unify_dataloader(test_filelist_dir)
                    # TODO
                    # prepare core set
                    DMAD.create_coreset(train_dataloader, NG_dataloader)
                    i_auroc, p_auroc, pro_auroc = DMAD.train(train_dataloader, test_dataloaders,
                                                                  setting="unify", NG_data=cycle(NG_dataloader))

                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "instance_auroc": i_auroc,  # auroc,
                        "full_pixel_auroc": p_auroc,  # full_pixel_auroc,
                        "anomaly_pixel_auroc": pro_auroc,
                    }
                )

                for key, item in result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:3.3f}".format(key, item))
                        # Store all results and mean scores to a csv-file.
                        result_metric_names = list(result_collect[-1].keys())[1:]
                        result_dataset_names = [results["dataset_name"] for results in result_collect]
                        result_scores = [list(results.values())[1:] for results in result_collect]
                        my_utils.compute_and_store_final_results(
                            exp_save_path,
                            result_scores,
                            column_names=result_metric_names,
                            row_names=result_dataset_names,
                        )

            elif mode == "test":
                # traverse sub datasets
                # TODO
                DMAD.loading_coreset()
                for ds_count, sub_dataset in enumerate(sub_datasets, start=1):
                    LOGGER.info(
                        "Evaluating dataset [{}] ({}/{})...".format(
                            sub_dataset,
                            ds_count,
                            len(sub_datasets),
                        )
                    )
                    test_dataloader = get_sub_dataloader(sub_dataset, test_filelist_dir)

                    DMAD.gen_anomap(test_dataloader, dataset_name, dataset_path,
                                         sub_dataset, time_str)

        LOGGER.info("\n\n-----\n")


if __name__ == "__main__":
    run()