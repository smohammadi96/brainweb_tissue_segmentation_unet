import cv2
import numpy as np
import glob
import os

from pyrsistent import v


def convert_multi_label(image_matrix):
    black = 0
    five = image_matrix
    print(np.unique(image_matrix))

    conv_255 = np.where((five == 255), five, black)
    conv_255 = np.where((conv_255 != 255), conv_255, 255)

    conv_25 = np.where((five == 25), five, black)
    conv_25 = np.where((conv_25 != 25), conv_25, 255)

    conv_51 = np.where((five == 51), five, black)
    conv_51 = np.where((conv_51 != 51), conv_51, 255)

    conv_76 = np.where((five == 76), five, black)
    conv_76 = np.where((conv_76 != 76), conv_76, 255)

    conv_204 = np.where((five == 204), five, black)
    conv_204 = np.where((conv_204 != 204), conv_204, 255)

    final_conv = conv_255 + conv_25 + conv_76 + conv_51 + conv_204
    
    return final_conv


for folder in glob.glob(os.path.join("sagital\\sagital_MS_dataset_label", "*")):
    for png_file in glob.glob(os.path.join(folder, "*.png")):
        img = cv2.imread(png_file)
        if len(np.unique(img))==11:
            out_mask_path = os.path.join("sagital\\sagital_MS_dataset_mask", folder.split('\\')[-1], png_file.split('\\')[-1])
            binary_image = convert_multi_label(img)
            cv2.imwrite(out_mask_path, binary_image)