import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from constants.dataset_constants import *
""" 
Logging functions in train, validation, test phase
"""
def log_configuration(phase, model_name, train_config, device_name):
    """
    Logging final settings for train, validation and test phase:
        device, learning rate, metrics,...
    """
    if phase == "train":
        learning_rate = train_config["learning_rate"]
        batch_size = train_config["batch_size"]
        epochs = train_config["epochs"]
        print(
            f"Train model [{model_name}] on CONLL-2003 train dataset({TRAIN_NUM_SAMPLES} samples) : device = {device_name}, learning rate = {learning_rate}, batch size = {batch_size}, total epochs = {epochs} "
        )

    elif phase == "validation":
        print(
            f"Validate model [{model_name}] on CONLL-2003 validation dataset({VALIDATION_NUM_SAMPLES} samples) : device = {device_name}, metrics = f1"
        )

    elif phase == "test":
        print(
            f"Test model [{model_name}] on CONLL-2003 test dataset({TEST_NUM_SAMPLES} samples) : device = {device_name}, metrics = f1, precision, recall"
        )


def log_epoch_result(epoch, total_eppochs, epoch_loss, f1, learning_rate, epoch_time):
    print(
        f"Epoch {epoch+1}/{total_eppochs}: train_loss: {epoch_loss:.2f} - validation_f1: {f1:.2f} - lr: {learning_rate} - time: {epoch_time:.0f}s"
    )


def log_saving_result(saving_path, saving_best_path):
    print(f"Successfully save model to {saving_path}")
    print(f"Successfully save model to {saving_best_path}")


def log_test_result(f1, precision, recall, report):
    print(f"Result: F1: {f1:.2f} - Precision: {precision:.2f} - Recall: {recall:.2f}")
    print(report)
