from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

   
def seg_gen_train(img_path, msk_path, img_size, batch_size):
    datagenerator = ImageDataGenerator(rescale=1./255,
                                       rotation_range = 40,
                                       horizontal_flip = True,
                                       vertical_flip = True)

    gen_params = dict(target_size=img_size, class_mode=None, color_mode='grayscale', batch_size=batch_size)
    # img_generator = datagenerator.flow_from_directory(img_path, **gen_params)
    # msk_generator = datagenerator.flow_from_directory(msk_path, **gen_params)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    img_generator = datagen.flow_from_directory(img_path, **gen_params)
    msk_generator = datagen.flow_from_directory(msk_path, **gen_params)
    s = 0
    for i in range(len(img_generator)):
        s = s + len(img_generator[i])
    # Loading a sample image 
    img = load_img('folder1\\train\\mask\\img\\20.png') 
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1, ) + x.shape) 
   
    # Generating and saving 5 augmented samples 
    # using the above defined parameters. 
    i = 0
    for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir ='preview', 
                          save_prefix ='image_mask', save_format ='jpeg'):
        i += 1
        if i > 5:
            break
    return zip(img_generator, msk_generator)

def seg_gen_test(img_path, msk_path, img_size, batch_size):
    datagenerator = ImageDataGenerator(rescale=1./255)
    gen_params = dict(target_size=img_size, class_mode=None, color_mode='grayscale', batch_size=batch_size)
    img_generator = datagenerator.flow_from_directory(img_path, **gen_params)
    msk_generator = datagenerator.flow_from_directory(msk_path, **gen_params)
    return zip(img_generator, msk_generator)
