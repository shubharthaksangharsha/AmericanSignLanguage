import os
import string
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def create_y_labels(test_data):
    y_labels = []
    num_of_examples = len(test_data.filenames)
    num_of_generator_calls = math.ceil(num_of_examples / (1.0 * 32))
    for i in range(0, int(num_of_generator_calls)):
        y_labels.extend(np.array(test_data[i][1]))
    return y_labels

def preprocess(img):
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.
    return img

def preprocess_filename(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.
    return img

def create_folder():
    if not os.path.exists("./Data"):
        os.mkdir("./Data/")
    for i in string.ascii_uppercase:
        if not os.path.exists(f"./Data/{i}"):
            os.makedirs(f"./Data/{i}")
def predict_and_plot(model, filename, classes):
    img = preprocess(filename)
    plt.figure(figsize=(10, 7))
    plt.axis(False)
    preds = model.predict(tf.expand_dims(img, axis = 0)).argmax()
    pred_class = classes[preds]
    plt.title(pred_class)
    plt.imshow(img);

def save_model(model, no=1):
    if not os.path.exists("./Models"):
        os.mkdir("./Models")
    if not os.path.exists(f"./Models/model{no}"):
        os.mkdir(f"./Models/model{no}")
    name = f"./Models/model{no}/model{no}.h5"
    model.save(name)


# Plot the validation and training data separately
def plot_loss_curves(history, no=1):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  if not os.path.exists("./Models"):
      os.mkdir("./Models")
  if not os.path.exists(f"./Models/model{no}"):
      os.mkdir(f"./Models/model{no}")

  loss_filename = f"./Models/model{no}/loss.png"
  accuracy_filename = f"./Models/model{no}/accuracy.png"
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.figure()
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.savefig(loss_filename);


  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  plt.savefig(accuracy_filename);


# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=True, no=1):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).

    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig(f"./Models/model{no}/confusion_matrix{no}.png")



if __name__ == "__main__":
    pass