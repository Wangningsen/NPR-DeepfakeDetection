import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
    'CVPJ_Val': {
        # 注意: dataroot 是上一级目录, 下面再套一层 "ours"
        'dataroot': '/data1/nwang60/datasets/NPR_ValForCVPJ/',
        'no_resize': False,  # 你的图本来就是 256×256 的话可以设 True
        'no_crop': True,
    },
}
multiclass = [0]

opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
model = resnet50(num_classes=1)
from collections import OrderedDict
from copy import deepcopy

ckpt = torch.load(opt.model_path, map_location="cpu")

# 1) 先取出真正的模型参数字典
state_dict = ckpt.get("model", ckpt)

# 2) 去掉可能的前缀（比如 "module." 或 "model."）
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("module."):
        new_k = k[len("module."):]
    elif k.startswith("model."):
        new_k = k[len("model."):]
    else:
        new_k = k
    new_state_dict[new_k] = deepcopy(v)

# 3) 再喂给 ResNet
model.load_state_dict(new_state_dict, strict=True)
print("Loaded weights from", opt.model_path)

model.cuda()
model.eval()

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt)
        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

