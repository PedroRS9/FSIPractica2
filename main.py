from config import img_width, img_height, batch_size, num_epochs, test_directory, train_directory, validation_directory, model_name
from data_loader import prepare_data
from model_builder import construct_keras_model
import os
from keras.models import load_model
from keras.callbacks import EarlyStopping
from plotting import plot_training_history, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np

test_generator, train_generator, validation_generator = prepare_data(
    img_width,
    img_height,
    batch_size,
    test_directory, train_directory, validation_directory
)
if os.path.isfile(model_name):
    model = load_model(model_name)
    print("Model loaded")
else:
    model = construct_keras_model(img_width, img_height)

# compile the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train the model
history_of_train = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)
# save model
model.save(model_name)

plot_training_history(history = history_of_train, save_path="output/training_history.png")

score = model.evaluate(test_generator)
print(f'Precisión: {score[1]}')
print(f"Test Loss: {score[0]}")

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
conf_mat = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(confusion_matrix=conf_mat, labels=list(train_generator.class_indices.keys()), save_path="output/confusion_matrix.png")