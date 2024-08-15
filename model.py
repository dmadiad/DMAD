"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.cluster import KMeans

import common
import metrics

from coreset_sampler import ApproximateGreedyCoresetSampler, FaissNN
from anomap_util import create_anomap_savepath, save_anomap

from run import LOGGER

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class Cross_Attn(torch.nn.Module):
    def __init__(self, input_dim):
        super(Cross_Attn, self).__init__()
        self.n_dim = input_dim
        self.k_linear = torch.nn.Linear(self.n_dim, self.n_dim)
        self.v_linear = torch.nn.Linear(self.n_dim, self.n_dim)
        self.apply(init_weight)

    def forward(self, x, y):
        q = y.view(-1, 1, self.n_dim)
        k = self.k_linear(x).view(-1, 1, self.n_dim)
        v = self.v_linear(x).view(-1, 1, self.n_dim)

        q_k = torch.matmul(q, torch.transpose(k, -1, 1))
        q_k = torch.softmax(q_k, dim=-1)
        q_k_v = torch.matmul(q_k, v)
        return q_k_v.view(-1, self.n_dim)


class knowledge_Aug_Module(torch.nn.Module):
    def __init__(self, input_dim):
        super(Normality_Aug_Module, self).__init__()
        self.cross_attn = Cross_Attn(input_dim)
        self.norm = torch.nn.BatchNorm1d(input_dim)
        self.ffn = torch.nn.Linear(input_dim, input_dim)
        self.apply(init_weight)

    def forward(self, x, y):
        h = self.cross_attn(x, y)
        return self.norm(x - y + h)


class DisModule(torch.nn.Module):
    def __init__(self, channels):
        super(DisModule, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.BatchNorm1d(channels),
            torch.nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        h = self.module(x)
        return x + h


class MLP(torch.nn.Module):
    """
    params:
    in_planes:
    n_layers:
    hidden:
    """
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(MLP, self).__init__()

        self.hidden = hidden
        self.in_planes = in_planes

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()

        for i in range(n_layers-1):
            if i == 0:
                self.body.add_module('block%d' % (i + 1),
                    torch.nn.Sequential(
                      torch.nn.Linear(in_planes, _hidden),
                      torch.nn.BatchNorm1d(_hidden),
                      torch.nn.LeakyReLU(0.2)
                    ))
            else:
                self.body.add_module('block%d' % (i + 1),
                    DisModule(_hidden)
                )

        self.tail = torch.nn.Linear(_hidden, 1, bias=False)

        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()

        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn",
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:

    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1

class DMAD(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(DMAD, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1,
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj

        if self.pre_proj > 0:
            # TODO
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)

        # TODO
        self.knowledge_aug = Knowledge_Aug_Module(self.target_embed_dimension)
        self.knowledge_aug.to(self.device)
        self.knowledge_aug_opt = torch.optim.AdamW(self.knowledge_aug.parameters(), lr * .1)

        # MLP
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.mlp = MLP(self.target_embed_dimension * 3, n_layers=dsc_layers, hidden=dsc_hidden)
        self.mlp.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.mlp.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None


    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)


    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # transfer to [b, h//patch_size, w//patch_size, channels, patch_size, patch_size]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            # transfer to [b, channels, patch_size, patch_size, h//patch_size, w//patch_size]
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            # transfer to [b * channels * patch_size * patch_size, h//patch_size, w//patch_size]
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            # transfer to [b * channels * patch_size * patch_size, new_h//patch_size, new_w//patch_size]
            _features = _features.squeeze(1)
            # transfer to [b, channels, patch_size, patch_size, new_h//patch_size, new_w//patch_size]
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            # transfer to [b, new_h//patch_size, new_w//patch_size, channels, patch_size, patch_size]
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            # transfer to [b, new_h//patch_size * new_w//patch_size, channels, patch_size, patch_size]
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        # transfer to [b * new_h//patch_size * new_w//patch_size, channels, patch_size, patch_size]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling


        return features, patch_shapes


    def paired_feature_creator(self, t_features, f_features, patch_size, NG_masks):

        # get batch size
        batch_size = len(NG_masks)

        patch_size = patch_size[0]

        # transfer to [b, new_h//patch_size * new_w//patch_size, channels, patch_size, patch_size]
        t_features = t_features.reshape(batch_size, -1, t_features.shape[-1])
        f_features = f_features.reshape(batch_size, -1, f_features.shape[-1])

        filtered_NG_features = []
        filtered_normal_features = []

        for idx in range(batch_size):
            cur_NG_mask = NG_masks[idx].squeeze()
            cur_t_feature = t_features[idx]
            cur_f_feature = f_features[idx]
            positions = torch.nonzero(cur_NG_mask)
            for pos in positions:
                filtered_NG_features.append(cur_f_feature[pos[0] * patch_size + pos[1]])
                filtered_normal_features.append(cur_t_feature[pos[0] * patch_size + pos[1]])

        return torch.stack(filtered_normal_features, dim=0), torch.stack(filtered_NG_features, dim=0)


    def filtered_NG_feats(self, features, patch_size, NG_masks):
        # get batch size
        batch_size = len(NG_masks)

        patch_size = patch_size[0]

        # transfer to [b, new_h//patch_size * new_w//patch_size, channels, patch_size, patch_size]
        features = features.reshape(batch_size, -1, features.shape[-1])

        filtered_NG_features = []

        for idx in range(batch_size):
            cur_NG_mask = NG_masks[idx].squeeze()
            cur_f_feature = features[idx]
            positions = torch.nonzero(cur_NG_mask)
            for pos in positions:
                filtered_NG_features.append(cur_f_feature[pos[0] * patch_size + pos[1]])

        return torch.stack(filtered_NG_features, dim=0)


    def save_anomap(self, dataloader, exp_path, obj_name):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        with tqdm.tqdm(dataloader, desc="Generating...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    image = data["image"]
                    image_paths = data['image_path']
                scores, masks, feats = self._predict(image)

                try:
                    # save anomap
                    for i, mask in enumerate(masks):
                        save_path = save_anomap(exp_path, image_paths[i], mask)
                        print('Anomap saved in ', save_path)
                except Exception as e:
                    print('Anomap save error', e)
                    raise e


    def gen_anomap(self, test_data, dataset_name, dataset_path, obj_name, time_str):
        # load trained models
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'mlp' in state_dict:
                self.mlp.load_state_dict(state_dict['mlp'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            # create anomap savepath
            exp_path = create_anomap_savepath(dataset_name, obj_name, dataset_path, time_str)
            # save anomap
            self.save_anomap(test_data, exp_path, obj_name)
        else:
            raise Exception('no trained models')


    def test(self, training_data, test_data):

        ckpt_path = os.path.join(self.ckpt_dir, "models.ckpt")
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        # Compute PRO score & PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )

        full_pixel_auroc = pixel_scores["auroc"]

        return auroc, full_pixel_auroc

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):

        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)


            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
                # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
                                            norm_segmentations)
        else:
            full_pixel_auroc = -1
            pro = -1

        return auroc, full_pixel_auroc, pro


    def eval_test_data(self, test_data, setting):
        if setting == "one_class":
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations,
                                                                          features, labels_gt, masks_gt)
        else:
            auroc, full_pixel_auroc, anomaly_pixel_auroc = 0., 0., 0.
            for t_data in test_data:
                scores, _, _, labels_gt, _ = self.predict(t_data)
                cur_auroc, cur_full_pixel_auroc, cur_anomaly_pixel_auroc = self._evaluate(t_data, scores,
                                                                                          [], [],
                                                                                          labels_gt, [])
                auroc += cur_auroc
                full_pixel_auroc += cur_full_pixel_auroc
                anomaly_pixel_auroc += cur_anomaly_pixel_auroc

            auroc /= len(test_data)
            full_pixel_auroc /= len(test_data)
            anomaly_pixel_auroc /= len(test_data)

        return auroc, full_pixel_auroc, anomaly_pixel_auroc


    # TODO
    def create_coreset(self, normal_data, ng_data):
        if os.path.exists('coreset.pt') and os.path.exists('ng_coreset.pt'):
            self.loading_coreset()
        else:
            if not os.path.exists('coreset.pt'):
                coreset_sampler = ApproximateGreedyCoresetSampler(percentage=0.02, device=self.device)
                """Computes and sets the support features"""
                _ = self.forward_modules.eval()

                def _image_to_features(input_image):
                    with torch.no_grad():
                        input_image = input_image.to(torch.float).to(self.device)
                        feats, _ =  self._embed(input_image)
                        return feats

                features = []
                with tqdm.tqdm(
                        normal_data, desc="Computing normal support features...", position=1, leave=False
                ) as data_iterator:
                    for data in data_iterator:
                        image = data["image"]
                        features.extend([x.unsqueeze(0) for x in _image_to_features(image)])

                self.coreset = coreset_sampler.run(features)

                torch.save(self.coreset.detach().cpu(), 'coreset.pt')

            else:
                self.coreset = torch.load('coreset.pt', map_location=self.device)

            if not os.path.exists('ng_coreset.pt'):
                """Computes and sets the support features"""
                _ = self.forward_modules.eval()

                def _image_to_features(input_image, mask):
                    with torch.no_grad():
                        input_image = input_image.to(torch.float).to(self.device)
                        mask = (mask != 0).to(torch.float).to(self.device)

                        feats, patch_shapes = self._embed(input_image)
                        resized_mask = F.interpolate(mask, size=patch_shapes[0], mode='bilinear',
                                                     align_corners=False)

                        feats = self.filtered_NG_feats(feats, patch_shapes[0], resized_mask)

                        return feats

                features = []
                with tqdm.tqdm(
                        ng_data, desc="Computing ng support features...", position=1, leave=False
                ) as data_iterator:
                    for data in data_iterator:
                        image = data["image"]
                        mask = data['mask']

                        features.append(_image_to_features(image, mask))

                ng_coreset = torch.cat(features, dim=0)

                additional_pseudo_ng_feats = []
                K = 1
                kmeans = KMeans(n_clusters=K, random_state=0).fit(ng_coreset.detach().cpu())
                centers = kmeans.cluster_centers_
                for center in centers:
                    center = torch.tensor(center, dtype=torch.float32).unsqueeze(0)
                    additional_pseudo_ng_feats.append(center + torch.normal(0, self.noise_std, (ng_coreset.shape[0]//K, 1536)))

                additional_pseudo_ng_feats = torch.cat(additional_pseudo_ng_feats, dim=0).to(self.device)

                chosen_file_list = []
                additional_domain_feats = []
                source_path = "./dtddataset/dtd/images/"
                for domain in os.listdir(source_path):
                    domain_filelist = os.listdir(os.path.join(source_path, domain))
                    idx = torch.randint(0, len(domain_filelist), (1,)).item()
                    chosen_file_list.append(os.path.join(source_path, domain, domain_filelist[idx]))

                dtd_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                def _dtd_to_features(input_image):
                    with torch.no_grad():
                        input_image = input_image.to(torch.float).to(self.device)
                        feats, _ =  self._embed(input_image)
                        return feats

                for image_path in chosen_file_list:
                    image = Image.open(image_path).convert("RGB")
                    image = dtd_transform(image).unsqueeze(0)
                    additional_domain_feats.append(_dtd_to_features(image))

                additional_domain_feats = torch.cat(additional_domain_feats, dim=0)
                additional_domain_feats = [x.unsqueeze(0) for x in additional_domain_feats]
                outdomain_sampler = ApproximateGreedyCoresetSampler(percentage=0.1, device=self.device)
                additional_domain_feats = outdomain_sampler.run(additional_domain_feats)

                self.ng_coreset = torch.cat([ng_coreset, additional_pseudo_ng_feats, additional_domain_feats], dim=0)

                torch.save(self.ng_coreset.detach().cpu(), 'ng_coreset.pt')
            else:
                self.ng_coreset = torch.load('ng_coreset.pt', map_location=self.device)

    # TODO
    def loading_coreset(self):
        assert os.path.exists('coreset.pt')
        assert os.path.exists('ng_coreset.pt')

        self.coreset = torch.load('coreset.pt', map_location=self.device)
        self.ng_coreset = torch.load('ng_coreset.pt', map_location=self.device)


    # TODO
    def find_nn_features(self, nn_finder, query_feature, is_normal_coreset=True):
        _, query_nns = nn_finder.run(query_feature.detach().cpu().numpy())

        query_nns = torch.from_numpy(query_nns).to(self.device)

        if is_normal_coreset:
            return torch.index_select(self.coreset, 0, query_nns.squeeze())
        else:
            return torch.index_select(self.ng_coreset, 0, query_nns.squeeze())


    def train(self, training_data, test_data, setting="one_class", NG_data=None):

        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'mlp' in state_dict:
                self.mlp.load_state_dict(state_dict['mlp'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            self.predict(training_data, "train_")

            return self.eval_test_data(test_data, setting)

        def update_state_dict():

            state_dict["mlp"] = OrderedDict({
                k:v.detach().cpu()
                for k, v in self.mlp.state_dict().items()})

            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        best_record = None

        for i_mepoch in range(self.meta_epochs):

            if NG_data is not None:
                self._train_mlp(training_data, NG_data)
            else:
                self._train_mlp(training_data)

            if i_mepoch == 0 or (i_mepoch+1) % 4 == 0:
                torch.cuda.empty_cache()
                # update_state_dict(state_dict)

                auroc, full_pixel_auroc, pro = self.eval_test_data(test_data, setting)

                self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
                self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
                self.logger.logger.add_scalar("pro", pro, i_mepoch)

                if best_record is None:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict()
                else:
                    if auroc > best_record[0]:
                        best_record = [auroc, full_pixel_auroc, pro]
                        update_state_dict()
                        # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                    elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                        best_record[1] = full_pixel_auroc
                        best_record[2] = pro
                        update_state_dict()
                        # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

                print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                      f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                      f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")

        torch.save(state_dict, ckpt_path)

        return best_record


    def _train_mlp(self, input_data, NG_data=None):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.mlp.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training mlp...")

        # TODO
        normal_nn_finder = FaissNN()
        normal_nn_finder.fit(self.coreset.detach().cpu().numpy())

        ng_nn_finder = FaissNN()
        ng_nn_finder.fit(self.ng_coreset.detach().cpu().numpy())

        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)

                    # _embed: feature extractor, pre_projection: feature adaptor
                    true_feats, patch_shapes = self._embed(img, evaluation=False)

                    # TODO
                    nn_true_feats = self.find_nn_features(normal_nn_finder, true_feats)
                    nn_ng_feats = self.find_nn_features(ng_nn_finder, true_feats, is_normal_coreset=False)

                    aug_true_parts = self.knowledge_aug(true_feats, nn_true_feats)
                    aug_ng_parts = self.knowledge_aug(true_feats, nn_ng_feats)

                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(true_feats)
                        aug_true_parts = self.pre_projection(aug_true_parts)
                        aug_ng_parts = self.pre_projection(aug_ng_parts)

                    true_feats = torch.cat([true_feats, aug_true_parts, aug_ng_parts], dim=1)

                    noise = torch.normal(0, self.noise_std, [true_feats.shape[0], true_feats.shape[1]//3])
                    noise = torch.cat([noise, noise, noise], dim=1).to(self.device)

                    fake_feats = true_feats + noise

                    if NG_data is not None:
                        cur_NG_data = next(NG_data)
                        NG_img, NG_img_mask = cur_NG_data["image"], cur_NG_data["mask"]

                        while len(NG_img) < len(img):
                            next_NG_data = next(NG_data)
                            next_NG_img, next_NG_img_mask = next_NG_data["image"], next_NG_data["mask"]
                            NG_img = torch.cat([NG_img, next_NG_img], dim=0)
                            NG_img_mask = torch.cat([NG_img_mask, next_NG_img_mask], dim=0)

                        NG_img = NG_img[:len(img)]
                        NG_img_mask = NG_img_mask[:len(img)]

                        NG_img = NG_img.to(torch.float).to(self.device)
                        NG_img_mask = (NG_img_mask != 0).to(torch.float).to(self.device)

                        f_feats, patch_shapes = self._embed(NG_img, evaluation=False)

                        # resize mask
                        resized_NG_mask = F.interpolate(NG_img_mask, size=patch_shapes[0], mode='bilinear',
                                                        align_corners=False)
                        # filtered ng feats
                        f_feats = self.filtered_NG_feats(f_feats, patch_shapes[0], resized_NG_mask)

                        # TODO
                        nn_true_feats = self.find_nn_features(normal_nn_finder, f_feats)
                        residual_withnormal_feats = f_feats - nn_true_feats

                        nn_ng_feats = self.find_nn_features(ng_nn_finder, f_feats, is_normal_coreset=False)
                        residual_withng_feats = f_feats - nn_ng_feats

                        if self.pre_proj > 0:
                            f_feats = self.pre_projection(f_feats)
                            residual_withnormal_feats = self.pre_projection(residual_withnormal_feats)
                            residual_withng_feats = self.pre_projection(residual_withng_feats)

                        f_feats = torch.cat([f_feats, residual_withnormal_feats, residual_withng_feats], dim=1)

                        fake_feats = torch.cat([fake_feats, f_feats])

                        scores = self.mlp(torch.cat([true_feats, fake_feats]))
                        true_scores = scores[:len(true_feats)]
                        fake_scores = scores[len(true_feats):-len(f_feats)]
                        true_fake_scores = scores[-len(f_feats):]

                        th = self.dsc_margin
                        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                        p_true_fake = (true_fake_scores.detach() < -th).sum() / len(true_fake_scores)

                        true_loss = torch.clip(-true_scores + th, min=0)
                        fake_loss = torch.clip(fake_scores + th, min=0)
                        true_fake_loss = torch.clip(true_fake_scores + th, min=0)

                        loss = true_loss.mean() + fake_loss.mean() + true_fake_loss.mean() * 10

                        self.logger.logger.add_scalar(f"p_true_fake", p_true_fake, self.logger.g_iter)
                    else:
                        scores = self.mlp(torch.cat([true_feats, fake_feats]))
                        true_scores = scores[:len(true_feats)]
                        fake_scores = scores[len(true_feats):]

                        th = self.dsc_margin
                        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)

                        # 让true score值大于0.5，fake score值小于-0.5
                        true_loss = torch.clip(-true_scores + th, min=0)
                        fake_loss = torch.clip(fake_scores + th, min=0)

                        loss = true_loss.mean() + fake_loss.mean()

                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()

                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

                if self.cos_lr:
                    self.dsc_schl.step()

                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()


        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)

                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()

        self.mlp.eval()

        # TODO
        normal_nn_finder = FaissNN()
        normal_nn_finder.fit(self.coreset.detach().cpu().numpy())

        ng_nn_finder = FaissNN()
        ng_nn_finder.fit(self.ng_coreset.detach().cpu().numpy())

        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True,
                                                 evaluation=True)

            # TODO
            nn_true_feats = self.find_nn_features(normal_nn_finder, features)
            nn_ng_feats = self.find_nn_features(ng_nn_finder, features, is_normal_coreset=False)

            aug_true_parts = self.knowledge_aug(true_feats, nn_true_feats)
            aug_ng_parts = self.knowledge_aug(true_feats, nn_ng_feats)

            if self.pre_proj > 0:
                features = self.pre_projection(features)
                aug_true_parts = self.pre_projection(aug_true_parts)
                aug_ng_parts = self.pre_projection(aug_ng_parts)

            features = torch.cat([features, aug_true_parts, aug_ng_parts], dim=1)

            patch_scores = image_scores = -self.mlp(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        # b,c,h,w -> b,c*k*k,patchsize,patchsize
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
