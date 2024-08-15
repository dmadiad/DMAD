import os
import numpy as np

def create_anomap_savepath(dataset_name, obj_name, dataset_path, time_str):

    assert os.path.exists(dataset_path), 'datasets path not exist!'

    # exp path
    exp_basepath = './anomap_res/'

    # join ds name
    exp_path = os.path.join(exp_basepath, dataset_name)
    # join time str
    exp_path = os.path.join(exp_path, time_str)
    # join object name
    exp_path = os.path.join(exp_path, obj_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    clss = os.listdir(os.path.join(dataset_path, obj_name, 'test'))
    for cls in clss:
        cls_path = os.path.join(exp_path, cls)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)

    return exp_path

def save_anomap(exp_path, img_path, anomap):
    assert os.path.exists(exp_path), 'experiment path not exist!'
    # get class path(good, NG_name1, ..., etc)
    if "test" in img_path:
        _, _, cls_path = img_path.partition('test/')
    else:
        _, _, cls_path = img_path.partition('val/')
    # split to class name and instance name
    cls_name, file_name = os.path.split(cls_path)
    # trans instance name
    basename = os.path.basename(file_name)
    name_without_extension = basename[:basename.rfind('.')]
    file_name = name_without_extension + '.npy'
    # obj name: prod name, cls name: class name
    anomap_savepath = os.path.join(exp_path, cls_name)
    # check if the save path exist
    assert os.path.exists(anomap_savepath), 'anomap save path not exist!'
    print(anomap_savepath, file_name)
    anomap_savepath = os.path.join(anomap_savepath, file_name)
    np.save(anomap_savepath, anomap)
    return anomap_savepath