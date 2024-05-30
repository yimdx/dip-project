import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.ndimage import minimum_filter
from dark import guide_filter, eps

def calculate_depth_map(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1] / 255.0
    value = hsv_image[:, :, 2] / 255.0
    sigma = 0.041337
    sigma_matrix = np.random.normal(0, sigma, (image.shape[0], image.shape[1]))
    depth_map = 0.121779 + 0.959710 * value - 0.780245 * saturation + sigma_matrix
    return depth_map

def haze_removal(image):
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = calculate_depth_map(input_image)
    local_min_depth = minimum_filter(depth_map, 15)
    radius = 8
    guided_depth = guide_filter(depth_map, local_min_depth, radius)
    flat_indices = np.argsort(guided_depth, axis=None)
    rows, cols = guided_depth.shape
    top_indices_flat = flat_indices[int(np.round(0.999 * rows * cols))::]
    top_indices = np.unravel_index(top_indices_flat, guided_depth.shape)
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2] / 255.0
    max_value_index = np.unravel_index(np.argmax(value_channel[top_indices], axis=None), value_channel.shape)
    input_image = input_image / 255.0
    atmospheric_light = input_image[max_value_index[0], max_value_index[1], :]
    beta = 1.0
    transmission = np.minimum(np.maximum(np.exp(-1 * beta * guided_depth), 0.1), 0.9)
    transmission_matrix = np.zeros(input_image.shape)
    transmission_matrix[:, :, 0] = transmission
    transmission_matrix[:, :, 1] = transmission
    transmission_matrix[:, :, 2] = transmission
    dehazed_image = atmospheric_light + (input_image - atmospheric_light) / transmission_matrix
    dehazed_image = dehazed_image - np.min(dehazed_image)
    dehazed_image = dehazed_image / np.max(dehazed_image)
    dehazed_image = dehazed_image * 255.0
    dehazed_image = dehazed_image.astype(np.uint8)
    bgr_image = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)
    return bgr_image

if __name__ == '__main__':
    image_path = 'example.jpg'
    input_image = cv2.imread(image_path)
    dehazed_image = haze_removal(input_image)
    cv2.imwrite("color.png", dehazed_image)
