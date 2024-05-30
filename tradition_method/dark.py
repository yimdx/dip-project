import cv2
import numpy as np
import math

eps = 1e-5


def dark_channel(image, window_size):
    b, g, r = cv2.split(image)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(image, dark):
    [h, w] = image.shape[:2]
    size = h*w
    numpx = int(max(math.floor(size/1000), 1))
    darkvec = dark.reshape(size)
    imvec = image.reshape(size, 3)

    indices = darkvec.argsort()
    indices = indices[size-numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
    atmo = atmsum / numpx
    return atmo


def trans_estimate(image, atmo, size=15, w=0.95):
    x = image / atmo
    t = 1 - w * dark_channel(x, size)
    return t


def guide_filter(image, guide, ksize):
    mean_i = cv2.blur(image, (ksize, ksize))
    mean_p = cv2.blur(guide, (ksize, ksize))
    corr_i = cv2.blur(np.multiply(guide, guide), (ksize, ksize))
    corr_ip = cv2.blur(np.multiply(guide, image), (ksize, ksize))
    var_i = corr_i - np.multiply(mean_i, mean_i)
    cov_ip = corr_ip - np.multiply(mean_i, mean_p)
    a = cov_ip / (var_i + eps)
    b = mean_p - np.multiply(a, mean_i)
    mean_a = cv2.blur(a, (ksize, ksize))
    mean_b = cv2.blur(b, (ksize, ksize))
    guide = np.multiply(mean_a, image) + mean_b
    return guide


def dcp_dehaze(image, omega=0.95, window_size=15, guided_filter_radius=60, guided_filter_epsilon=1e-5):
    global eps
    eps = guided_filter_epsilon
    dark = dark_channel(image, window_size)
    atmospheric_light = estimate_atmospheric_light(image, dark)
    trans = trans_estimate(image, atmospheric_light)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transmission_guidance = guide_filter(gray.astype(np.float32), trans.astype(np.float32), guided_filter_radius)

    dehazed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        dehazed_image[:, :, i] = (image[:, :, i].astype(np.float32) - atmospheric_light[0,i]) / np.maximum(transmission_guidance, 0.1) + atmospheric_light[0, i]
    return np.clip(dehazed_image, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread("example.jpg")
    ret = dcp_dehaze(image)
    cv2.imwrite("dcp.png", ret)
    cv2.waitKey()
