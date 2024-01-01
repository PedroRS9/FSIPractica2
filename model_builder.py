from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def construct_keras_model(img_width, img_height):
    '''
    Constructs the Keras model for the CNN
    :param img_width: width of the images
    :param img_height: height of the images
    :return: the Keras model
    '''
    keras_model = Sequential()
    keras_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    keras_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    keras_model.add(Flatten())
    # Capas fully connected
    keras_model.add(Dense (128, activation='relu'))
    keras_model.add(Dense(15, activation='softmax'))
    return keras_model