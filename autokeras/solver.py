from mlpm.solver import Solver
from mlpm.logger import StepLogger

import cv2


class LogsCallback(tf.keras.callbacks.Callback):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        self.logger.append(
            {'epoch': epoch, 'loss': logs['loss'], 'mae': logs['mae']})


class MNISTSolverExample(Solver):
    def __init__(self):
        sl = StepLogger()
        super().__init__()

    def train(self, data, config):
        super().train()
        (x_train, y_train), (x_test, y_test) = mnist.load_data(data)
        # Initialize the ImageClassifier.
        clf = ak.ImageClassifier(max_trials=3)
        # Search for the best model.
        clf.fit(x_train, y_train, callbacks=[LossAndErrorPrintingCallback()])
        # Evaluate on the testing data.
        print('Accuracy: {accuracy}'.format(
            accuracy=clf.evaluate(x_test, y_test)))


