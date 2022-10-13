import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse

import torch
from defaults.model_defaults import MODEL_MODE
from defaults.tag_dict import *
from defaults.train_defaults import EPOCHS, USE_GPU
from helpers.logger import log_configuration, log_test_result
from model import NERModel
from model_configs.configs import get_configs
from prepare.data import load_transform_data
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)


def train_validate(model_configs, train_config, state_dict, dataloader):
    """
    Function for help train phase to validate epoch's model on validation dataset:
    :model_configs, train_config, state_dict: given by train phase
    :dataloader: Dataloader instance of validation dataset
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

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        all_predicts = []
        true_tags = []
        for i, batch in enumerate(dataloader):
            best_paths = model.forward(batch)
            seq_lengths = batch["features"]["seq_lengths"]
            idxs_to_tags = lambda idxs: [IDX_TO_TAG[idx] for idx in idxs]
            for j in range(len(batch)):

                sample_predicts = best_paths[j][: seq_lengths[j]]

                sample_true_tags = batch["tags"][j][: seq_lengths[j]].long().tolist()

                all_predicts.append(idxs_to_tags(sample_predicts))
                true_tags.append(idxs_to_tags(sample_true_tags))

        f1 = f1_score(true_tags, all_predicts)
        return f1


def run(model_configs, weights_path, device_mode):
    """
    Validate chosen model with weights saved in weights_path.
    """
    device_name = "cpu"
    device = torch.device(device_name)
    if device_mode == 1:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

    if weights_path == "":
        weights_path = ROOT / "models" / model_configs["model_name"] / "best.w"
    else:
        weights_path = ROOT / "models" / model_configs["model_name"] / weights_path
    model = NERModel(
        model_configs["lstm"],
        model_configs["char_level_embedding"],
        model_configs["casing_embedding"],
        model_configs["crf"],
        device,
    ).to(device)

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    dataloader = load_transform_data(
        "test",
        use_char_level=model_configs["use_char_level_embedding"],
        use_casing=model_configs["use_casing_feature"],
    )

    log_configuration("test", model_configs["model_name"], None, device_name)

    with torch.no_grad():
        all_predicts = []
        true_tags = []
        for i, batch in enumerate(dataloader):
            best_paths = model.forward(batch)
            seq_lengths = batch["features"]["seq_lengths"]
            idxs_to_tags = lambda idxs: [IDX_TO_TAG[idx] for idx in idxs]
            for j in range(len(batch)):

                sample_predicts = best_paths[j][: seq_lengths[j]]
                sample_true_tags = batch["tags"][j][: seq_lengths[j]].long().tolist()

                all_predicts.append(idxs_to_tags(sample_predicts))
                true_tags.append(idxs_to_tags(sample_true_tags))

        f1 = round(f1_score(true_tags, all_predicts) * 100, 2)
        precision = round(precision_score(true_tags, all_predicts) * 100, 2)
        recall = round(recall_score(true_tags, all_predicts) * 100, 2)
        
        log_test_result(f1, precision, recall, classification_report(true_tags, all_predicts))



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-mode", type=int, default=MODEL_MODE, help="mode of model"
    )
    parser.add_argument("--weights", type=str, default="", help="weights file name")
    parser.add_argument(
        "--device-mode", type=int, default=USE_GPU, help="0: cpu - 1: cuda"
    )
    return parser.parse_args()


def main(opt):
    model_configs = get_configs(opt.model_mode)
    weights_path = opt.weights
    device_mode = opt.device_mode
    run(model_configs, weights_path, device_mode)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
