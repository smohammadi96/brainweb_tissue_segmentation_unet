from tensorflow import keras  
import tensorflow.keras.layers as layers


def UNet(in_channels, out_channels, n_levels, initial_features, n_blocks, IMAGE_HEIGHT, IMAGE_WIDTH):
   
    inputs = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    
    x = inputs
    
    skips_connections = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = layers.Conv2D(initial_features * 2 ** level, kernel_size=3, activation='relu', padding='same')(x)
        if level < n_levels - 1:
            skips_connections[level] = x 
            x = layers.MaxPool2D(2)(x) 
            
    for level in reversed(range(n_levels-1)): 
        x = layers.Conv2DTranspose(initial_features * 2 ** level, strides=2, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.Concatenate()([x, skips_connections[level]]) 
        for _ in range(n_blocks):
            x = layers.Conv2D(initial_features * 2 ** level, kernel_size=3, activation='relu', padding='same')(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-Level{n_levels}-Features{initial_features}')
