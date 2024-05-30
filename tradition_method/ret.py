import cv2
import numpy as np


def retinex(img, sigma_list=[15, 80, 250]):
    img = img.astype(np.float64)
    img_msr = msr(img, sigma_list)
    img_msr = clip_image(img_msr, 0.05, 0.95)
    img_msr = ((img_msr - np.min(img_msr)) / (np.max(img_msr) - np.min(img_msr))) * 255
    return img_msr.astype(np.uint8)


def ssr(img, sigma):
    L_img = cv2.GaussianBlur(img, (0, 0), sigma)
    ans = np.log(img + 1) - np.log(L_img + 1)
    return ans


def msr(img, sigma_list):
    tmp = np.zeros_like(img)
    for sigma in sigma_list:
        tmp += ssr(img, sigma)
    tmp /= len(sigma_list)
    return tmp


def clip_image(img, low_clip, high_clip):
    low_val, high_val = np.percentile(img, [low_clip * 100, high_clip * 100])
    img = np.clip(img, low_val, high_val)
    return img


if __name__ == '__main__':
    image = cv2.imread("example.jpg")
    result = retinex(image)
    cv2.imwrite("retinex_output.png", result)
