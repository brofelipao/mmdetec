import sys, os

from mmcv import Config
import os
import sys
import time
import matplotlib
import matplotlib.pylab as plt
from mmdet.datasets import build_dataset,build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet import __version__
from mmdet.utils import collect_env, get_root_logger
from mmcv.utils import get_git_hash
import torch
import os.path as osp
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets import CocoDataset
from mmcv.visualization import color_val
from mmdet.core import visualization as vis
import mmcv, torch
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


sys.path.append('mmdetection')

pasta_mmdetection=os.path.join(os.getcwd(),'mmdetection')
pasta_dataset=os.path.join(os.getcwd(),'dataset')
pasta_checkpoints=os.path.join(os.getcwd(),'checkpoints')
print('Pasta do mmdetection: ',pasta_mmdetection)
print('Pasta com o dataset: ',pasta_dataset)
print('Pasta com os checkpoints (*.pth): ',pasta_checkpoints)

plt.rcParams["axes.grid"] = False


def printToFile(linha='',arquivo='dataset/results.csv',modo='a'):
  original_stdout = sys.stdout # Save a reference to the original standard output
  with open(arquivo, modo) as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(linha)
    sys.stdout = original_stdout # Reset the standard output to its original value

print('======================================================')
print(' INICIANDO TESTE - INICIANDO TESTE - INICIANDO TESTE')
print('======================================================')

printToFile('ml,fold,groundtruth,predicted','dataset/counting.csv','w')

printToFile('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r','dataset/results.csv','w')
i=1
for selected_model in REDES:
    for f in np.arange(1,DOBRAS+1):
      print('------------------------------------------------------')
      print('-- TESTANDO A REDE ',selected_model,' NA DOBRA ',f)
      print('------------------------------------------------------')

      fold = 'fold_'+str(f)
      cfg = setCFG(selected_model=selected_model,data_root=pasta_dataset,classes=('Corn',),fold=fold)

      pth = os.path.join(cfg.data_root,(fold+'/MModels/%s/latest.pth'%(selected_model)))
      print('Usando o modelo aprendido: ',pth)
      resAP50 = testingModel(cfg=cfg,models_path=pth,show_imgs=False,save_imgs=True,num_model=i,fold=fold)
      printToFile(str(i)+'_'+selected_model + ','+fold+','+resAP50,'dataset/results.csv','a')
    i=i+1