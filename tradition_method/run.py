import cv2
import os
from tqdm import tqdm
from dark import dcp_dehaze
from color import haze_removal
from ret import retinex


def deal_images(haze_path, output_path, mode="dcp"):

    haze_file_list = [img_name for img_name in os.listdir(haze_path)]
    print(haze_file_list)
    for idx, img_name in enumerate(tqdm(haze_file_list)):
        haze = cv2.imread(os.path.join(haze_path, img_name))
        if mode == "dcp":
            output = dcp_dehaze(haze)  
        elif mode == "cap":
            output = haze_removal(haze)
        elif mode == "ret":
            output = retinex(haze)
        else:
            raise ValueError("no support mode")
        cv2.imwrite(os.path.join(output_path, img_name), output)


if __name__ == '__main__':
    # haze_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\hazy"
    # dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\dcp"
    # color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\color"
    # ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\O-haze\\ret"

    # haze_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\data\\simu"
    # dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\dcp"
    # color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\color"
    # ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\HazeRD\\ret"


    haze_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\hazy"
    dcp_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\dcp"
    color_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\color"
    ret_path = "C:\\Users\\15957\\Desktop\\2023\\dip\\tradition_method\\data\\SOTS\\outdoor\\ret"

    deal_images(haze_path, dcp_path, "dcp")
    deal_images(haze_path, color_path, "cap")
    deal_images(haze_path, ret_path, "ret")

