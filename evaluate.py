import sys
import cv2
import numpy as np
from PIL import Image

# import torch

sys.path.append('../')

# from model.patch_based_cnn import net_baesd_patch, patch_test_transform
from lib.processing_utils import get_file_list, FaceDection
from patch_generate import patch_generate,RandomCrop
from keras.models import load_model
model = load_model('densenet169_fine_tuned.best.hdf5')

def patch_cnn_test(image):
    img_Image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    seed_arr = np.arange(8)
    score = []
    for i in range(8):
        img_transform = RandomCrop(size=96, seed=seed_arr[i])
        try:
            img_patch = img_transform(img_Image)
            if True:
                img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
                # cv2.imshow("patch", img_patch_opencv)
                # cv2.waitKey(0)
                score.append(model.predict(img_patch_opencv))

        except Exception as e:
            print(e)

    print(score)

if __name__ == '__main__':
    test_dir = "/home/shicaiwei/data/liveness_data/CASIA-FASD/test/spoofing"
    label = 0
    pre_path = "../output/models/patch_fasd.pth"
    isface = True
    img = cv2.imread('MSU-MFSD/test/real/real_client001_android_SD_scene01_0.jpg')
    patch_cnn_test(img)