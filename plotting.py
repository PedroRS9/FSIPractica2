import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
def plot_training_history(history, save_path=None):
    # Extracción de datos
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Creación del gráfico de accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training')
    plt.plot(epochs, val_acc, 'r', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Creación del gráfico de loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training')
    plt.plot(epochs, val_loss, 'r', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)

def plot_confusion_matrix(confusion_matrix, labels, save_path=None):
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    display.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)