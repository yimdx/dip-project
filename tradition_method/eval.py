from zipfile import LargeZipFile
import cv2
import numpy as np
from os import path as osp
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab, deltaE_cie76
from tqdm import tqdm
from PIL import Image

root = osp.dirname(osp.abspath(__file__))
OHaze_root = osp.abspath(osp.join(root, '../data', 'O-Haze'))


def calculate_metrics(image_gt, image_pred):
    mse = np.mean((image_gt - image_pred) ** 2)
    ssim_value = ssim(image_gt, image_pred, data_range=1, channel_axis=2)
    psnr_value = psnr(image_gt, image_pred)
    # image_gt_lab = rgb2lab(image_gt)
    # image_pred_lab = rgb2lab(image_pred)
    ciede2000 = 0
    # ciede2000 = np.mean([deltaE_cie76(color_gt, color_pred) for color_gt, color_pred in zip(image_gt_lab.reshape(-1, 3), image_pred_lab.reshape(-1, 3))])
    return mse, ssim_value, psnr_value, ciede2000


def load_images(gt_path, eval_path):
    
    gt_file_list = [img_name for img_name in os.listdir(gt_path)]
    gt_images = []
    for idx, img_name in enumerate(tqdm(gt_file_list)):
        gt = cv2.imread(os.path.join(gt_path, img_name))
        gt_images.append(gt)

    eval_file_list = [img_name for img_name in os.listdir(eval_path)]
    eval_images = []
    last_file_pre = None
    for idx, img_name in enumerate(tqdm(eval_file_list)):
        if last_file_pre == img_name.split('_')[0]:
            continue
        last_file_pre = img_name.split('_')[0]
        eval = cv2.imread(os.path.join(eval_path, img_name))
        eval_images.append(eval)
   
    mse_list = []
    ssim_list = []
    psnr_list = []
    ciede2000_list = []

    for image_gt, image_pred in tqdm(zip(gt_images, eval_images)):
        mse, ssim_value, psnr_value, ciede2000 = calculate_metrics(image_gt, image_pred)
        mse_list.append(mse)
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        ciede2000_list.append(ciede2000)

    average_mse = np.mean(mse_list)
    average_ssim = np.mean(ssim_list)
    average_psnr = np.mean(psnr_list)
    average_ciede2000 = np.mean(ciede2000_list)

    print(f"Average MSE: {average_mse}")
    print(f"Average SSIM: {average_ssim}")
    print(f"Average PSNR: {average_psnr}")
    print(f"Average CIEDE2000: {average_ciede2000}")


# gt_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\hazy"
# dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\dcp"
# color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\color"
# ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\ret"

hazerd_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\data\\img"
dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\dcp"
color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\color"
ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\ret"


def eval_dcp_ohaze():
    gt_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\gt"
    dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\dcp"
    load_images(gt_path, dcp_path)


def eval_cap_ohaze():
    gt_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\gt"
    color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\color"
    load_images(gt_path, color_path)


def eval_ret_ohaze():
    gt_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\gt"
    ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\ret"
    load_images(gt_path, ret_path)


def eval_dcp_hazerd():
    hazerd_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\data\\img"
    dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\dcp"
    load_images(hazerd_path, dcp_path)


def eval_cap_hazerd():
    hazerd_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\data\\img"
    color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\color"
    load_images(hazerd_path, color_path)


def eval_ret_hazerd():
    hazerd_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\data\\img"
    ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\ret"
    load_images(hazerd_path, ret_path)


def eval_dcp_sots():
    haze_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\gt"
    dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\dcp"
    load_images(haze_path, dcp_path)


def eval_cap_sots():
    haze_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\gt"
    color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\color"
    load_images(haze_path, color_path)


def eval_ret_sots():
    haze_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\gt"
    ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\ret"
    load_images(haze_path, ret_path)






if __name__ == '__main__':
    # print("on ohaze dataset")
    # print("dcp:")
    # eval_dcp_ohaze()
    # print("cap:")
    # eval_cap_ohaze()
    # print("ret:")
    # eval_ret_ohaze()
    # print("on hazerd dataset")
    # print("dcp:")
    # eval_dcp_hazerd()
    # print("cap:")
    # eval_cap_hazerd()
    # print("ret:")
    # eval_ret_hazerd()
    print("on sots")
    print("dcp:")
    eval_dcp_sots()
    print("cap:")
    eval_cap_sots()
    print("ret:")
    eval_ret_sots()
