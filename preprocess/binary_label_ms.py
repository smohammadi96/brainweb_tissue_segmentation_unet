import cv2
import numpy as np
import glob
import os


def convert_multi_label(image_matrix):
    black = 0
    one = image_matrix
    two = image_matrix
    three = image_matrix

    conv_204 = np.where((image_matrix != 255), image_matrix, 255)
    MS_lesion = np.where((conv_204 == 255), conv_204, 0)

    conv_25 = np.where((one != 25), one, 255)
    CSF = np.where((conv_25 == 255), conv_25, 0)

    conv_51 = np.where((two != 51), two, 255)
    GM = np.where((conv_51 == 255), conv_51, 0)

    conv_76 = np.where((two != 76), two, 255)
    WM = np.where((conv_76 == 255), conv_76, 0)
    
    return MS_lesion, CSF, GM, WM

folder = "ms_label_png\\out\\out_severe"
output = "ms_label_png\\binary_label\\severe_binary_label"
i = 0
for png_file in glob.glob(os.path.join(folder, "*.png")):
    img = cv2.imread(png_file)
    if len(np.unique(img))==11:
        png_name = png_file.split("\\")[-1]
        MS_lesion, CSF, GM, WM = convert_multi_label(img)
        cv2.imwrite(os.path.join(output, "MS", png_name), MS_lesion)
        cv2.imwrite(os.path.join(output, "CSF", png_name), CSF)
        cv2.imwrite(os.path.join(output, "GM", png_name), GM)
        cv2.imwrite(os.path.join(output, "WM", png_name), WM)
