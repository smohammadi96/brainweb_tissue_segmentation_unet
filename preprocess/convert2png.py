import os
import glob

import cv2
import numpy as np


def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv

def convert_multi_label(image_matrix):
    black = 0
    one = image_matrix
    two = image_matrix
    conv_23 = np.where((image_matrix == 23), image_matrix, black)
    conv_46 = np.where((one == 46), one, black)
    conv_69 = np.where((two == 69), two, black)
    final_conv = conv_23 + conv_46 + conv_69
    
    return final_conv


im = cv2.imread("200.png"))
cv2.imwrite("2.png", convert_binary(im))
print(np.unique(convert_binary(im)))



for sub_folder in glob.glob(".\BrainDatabase\BrainDatabase\*"):

    if not os.path.exists(os.path.join("binary_mask_new", sub_folder.split("\\")[-1])):
        os.mkdir(os.path.join("binary_mask_new", sub_folder.split("\\")[-1]))

    for folder in glob.glob(os.path.join(sub_folder, "*")):

        if folder.split("\\")[-1] in ["csf", "wm", "gm"]:
            if not os.path.exists(os.path.join("binary_mask_new", sub_folder.split("\\")[-1] , folder.split("\\")[-1])):
                os.mkdir(os.path.join("binary_mask_new", sub_folder.split("\\")[-1], folder.split("\\")[-1]))

            for png_file in glob.glob(os.path.join(folder, "*.png")):
                img = cv2.imread(png_file)
                binary_image = convert_binary(img, 125, folder.split("\\")[-1])
                png_path_output = os.path.join("binary_mask_new", sub_folder.split("\\")[-1],
                                               folder.split("\\")[-1], png_file.split("\\")[-1])
                print(png_path_output)
                cv2.imwrite(png_path_output, binary_image)
        else:
            print(folder)