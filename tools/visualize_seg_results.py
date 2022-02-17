from detectron2 import data
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json


if __name__=="__main__":

    data_path = '/home/nstarli/Mask2Former/work_dirs/sceneflow_vanilla_disp192/inference/sem_seg_predictions.json'
    with open(data_path, 'r') as f:
        results = json.load(f)
    results_list_full = []
    for res in results:
        