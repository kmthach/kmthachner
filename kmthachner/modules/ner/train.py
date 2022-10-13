import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import val as validate
from defaults.model_defaults import MODEL_MODE
from defaults.train_defaults import *
from helpers.callbacks import CustomCallback
from helpers.logger import log_configuration
from model import NERModel
from model_configs.configs import get_configs
from prepare.data import load_transform_data

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

def train(model_configs, train_config):
    """
    Create the model base on model config dict then train
    the model with settings in train config dict.
    """
    device_name = "cpu"
    device = torch.device(device_name)
    if train_config["device_mode"] == 1:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

    model = NERModel(
        model_configs["lstm"],
        model_configs["char_level_embedding"],
        model_configs["casing_embedding"],
        model_configs["crf"],
        device,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=train_config["learning_rate"])

    callback = CustomCallback(
        model_name=model_configs["model_name"],
        optimizer=optimizer,
        epochs=train_config["epochs"],
        factor=train_config["factor"],
        decay_patience=train_config["decay_patience"],
        mode=train_config["mode"],
        es_patience=train_config["es_patience"],
    )

    train_dataloader = load_transform_data(
        "train",
        use_char_level=model_configs["use_char_level_embedding"],
        use_casing=model_configs["use_casing_feature"],
    )

    validation_dataloader = load_transform_data(
        "validation",
        use_char_level=model_configs["use_char_level_embedding"],
        use_casing=model_configs["use_casing_feature"],
    )
    num_steps = len(train_dataloader)

    log_configuration("train", model_configs["model_name"], train_config, device_name)
    log_configuration(
        "validation", model_configs["model_name"], train_config, device_name
    )
    for epoch in range(train_config["epochs"]):
        epoch_loss = 0
        model.train()
        callback.reset_state()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.fit_and_compute_loss(batch)
            epoch_loss += loss
            loss.backward()
            clip_grad_norm_(model.parameters(), train_config["gradient_clipping"])
            optimizer.step()

        epoch_loss = epoch_loss / num_steps

        model_state_dict = model.state_dict()
        f1 = validate.train_validate(
            model_configs, train_config, model_state_dict, validation_dataloader
        )
        learning_rate = optimizer.param_groups[0]["lr"]

        stop_value = callback.step(
            epoch, epoch_loss, f1, learning_rate, model_state_dict
        )

        if stop_value == 1:
            break

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-mode", type=int, default=MODEL_MODE, help="mode of model"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, help="learning rate"
    )
    parser.add_argument(
        "--device-mode", type=int, default=USE_GPU, help="0: cpu - 1: cuda"
    )
    return parser.parse_args()

def main(opt):
    # Create model config and train config from parser and defaults
    model_config = get_configs(opt.model_mode)
    train_config = {}
    train_config["gradient_clipping"] = GRADIENT_CLIPPING
    train_config["factor"] = FACTOR
    train_config["decay_patience"] = DECAY_PATIENCE
    train_config["mode"] = MODE
    train_config["es_patience"] = ES_PATIENCE
    train_config["epochs"] = opt.epochs
    train_config["batch_size"] = opt.batch_size
    train_config["learning_rate"] = opt.learning_rate
    train_config["device_mode"] = opt.device_mode

    train(model_config, train_config)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
