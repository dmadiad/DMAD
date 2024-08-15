['']
# datasets config
dataset_name = "MVTec"
dataset_path = "./datasets/mvtec_anomaly_detection/"
# # VisA
# sub_datasets = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
#                 "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
# # MVTec
sub_datasets = ["bottle", "cable", "capsule", "carpet", "grid",
                "hazelnut", "leather", "metal_nut", "pill", "screw",
                "tile", "toothbrush","transistor", "wood", "zipper"]

# setting = "one_class"
setting = "unify"

# mode: train (unified), train_ng (unified semi-supervised), test
# mode = "train"
mode = "train_ng"
# mode = "test"

# train ng only
ng_nums = 10

image_size = 256

# training epoch
gpu_id = [0]
# random seed
seed = 0
meta_epochs = 40
gan_epochs = 4
batch_size = 32
test_batch_size=4

# net param
backbone_names = ["wideresnet50"]
layers_to_extract_from = ["layer2", "layer3"]
pretrain_embed_dimension = 1536
target_embed_dimension = 1536
patchsize = 3
embedding_size = 256
noise_std = 0.015
dsc_hidden = 1024
dsc_layers = 2
dsc_margin = .5
pre_proj = 1
dsc_lr = 0.0002
cos_lr = False
train_backbone = False
mix_noise = 1
auto_noise = .0
aed_meta_epochs = 1
proj_layer_type = 0