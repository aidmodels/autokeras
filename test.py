import autokeras as ak
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        self.logger.append({'epoch': epoch, 'loss': logs['loss'], 'mae':logs['mae']})

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data(
    path='/home/xzyao/.autoai/.aid/datasets/MNIST/mnist.npz')
# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train, callbacks=[LossAndErrorPrintingCallback()])
# Evaluate on the testing data.
print('Accuracy: {accuracy}'.format(
    accuracy=clf.evaluate(x_test, y_test)))
