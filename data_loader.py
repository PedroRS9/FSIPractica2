from keras.preprocessing.image import ImageDataGenerator


def prepare_data(img_width, img_height, batch_size, test_directory, train_directory, validation_directory):
    '''
    Prepares the data for the CNN
    :param img_width: the width of the images
    :param img_height: the height of the images
    :param batch_size: the batch size
    :param test_directory: the directory of the test data
    :param train_directory: the directory of the train data
    :param validation_directory: the directory of the validation data
    :return: test, train and validation generators
    '''
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical"
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical"
    )
    return test_generator, train_generator, validation_generator
