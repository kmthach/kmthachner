import sys
from pathlib import Path



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from defaults.train_defaults import BEST_FILE_NAME
from .logger import log_epoch_result, log_saving_result

import time
from datetime import datetime
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomCallback(object):
    """
    A Callback for Learning Rate Scheduler,
    Early Stopping, Saving Model,
    Logging, ...
    """
    def __init__(
        self, model_name, epochs, optimizer, factor, decay_patience, mode, es_patience
    ):
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode=mode, factor=factor, patience=decay_patience
        )
        self.model_name = model_name
        self.epochs = epochs
        self.es_patience = es_patience
        self.begin_time = time.time()
        self.best_f1 = -1
        self.bad_epochs = 0

    def step(self, epoch, epoch_loss, f1, learning_rate, model_state_dict):
        epoch_time = time.time() - self.begin_time
        f1 = round(f1 * 100, 2)
        log_epoch_result(epoch, self.epochs, epoch_loss, f1, learning_rate, epoch_time)

        self.lr_scheduler.step(f1)

        if f1 > self.best_f1:
            self.bad_epochs = 0
            self.best_f1 = f1
            file_name = str(datetime.now()) + " f1: " + str(f1) + ".w"
            best_file_name = BEST_FILE_NAME
            saving_path = ROOT / "models" / self.model_name / "exp" / file_name
            saving_best_path = (
                ROOT / "models" / self.model_name / "exp" / best_file_name
            )
            torch.save(model_state_dict, saving_path)
            torch.save(model_state_dict, saving_best_path)
            log_saving_result(saving_path, saving_best_path)
        else:
            self.bad_epochs += 1

        if self.bad_epochs > self.es_patience:
            return 1
        else:
            return 0

    def reset_state(self):
        self.begin_time = time.time()
