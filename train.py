import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import models.Unet as Unet
import utils.datagenerator
from utils.dataloader import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Batch Size train")
    parser.add_argument("--BS_train", help="Batch Size train")
    parser.add_argument("--BS_test", help="Batch Size test")
    parser.add_argument("--img_h", help="image height")
    parser.add_argument("--img_w", help="image width")
    parser.add_argument("--epochs", help="epochs")
    parser.add_argument("--model_name", help="model name")
  
    args = parser.parse_args()
    print("dataset:", args.dataset)    
    print("BS_train:", args.BS_train)    
    print("BS_test", args.BS_test)    
    print("img_h:", args.img_h)    
    print("img_w:", args.img_w)    
    print("epochs:", args.epochs)    
    
    # Load dataset (images and labels)
    data_loader = DataLoader(args.dataset)
    images, labels = data_loader.read_data()

    # Declare parameters
    BS_train = int(args.BS_train)
    BS_test = int(args.BS_test)
    img_h = int(args.img_h)
    img_w = int(args.img_w)
    img_size = (img_h, img_w)
    num_train = len(images)
    epochs = int(args.epochs)
    
    # Start training ...
    print("training ...")
    model = Unet.UNet(in_channels = 1, out_channels = 1, n_levels = 4, initial_features = 32, n_blocks = 2, IMAGE_HEIGHT = img_h, IMAGE_WIDTH = img_w)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    model.summary()
    model.fit(images, labels,
              batch_size= 4,
              epochs=epochs)

    # save model output
    model.save(f'{args.model_name}_{img_h}_{img_w}_{epochs}.h5')
    
    # Plot loss and accuracy during training
    acc = model.history.history['accuracy']
    loss = model.history.history['loss']
    epochs_range = [i for i in range(40)]
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.show()

if __name__ == "__main__":
    main()

