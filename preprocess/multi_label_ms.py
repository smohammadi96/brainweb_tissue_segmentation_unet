import cv2
import numpy as np
import glob
import os


def convert_multi_label(image_matrix):
    black = 0
    one = image_matrix
    two = image_matrix

    conv_204 = np.where((image_matrix == 255), image_matrix, 0)
    conv_25 = np.where((one == 25), one, black)
    conv_51 = np.where((two == 51), two, black)
    conv_76 = np.where((two == 76), two, black)
    final_conv = conv_204 + conv_25 + conv_51 + conv_76
    
    return final_conv


folder = "coronal\coronal_MS_dataset_label\\severe"
output = "coronal\coronal_MS_dataset_label\\out_severe_ms_multi_label"
i = 0
for png_file in glob.glob(os.path.join(folder, "*.png")):
    img = cv2.imread(png_file)
    if len(np.unique(img))==11:
        png_name = png_file.split('\\')[-1]
        binary_image = convert_multi_label(img)
        cv2.imwrite(os.path.join(output, png_name), binary_image)
