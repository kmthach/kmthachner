
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from api.api_defaults import *
from modules.defaults.tag_dict import IDX_TO_TAG
from modules.model_configs.configs import get_configs
from modules.ner.model import NERModel
from modules.prepare.data import load_from_array
from uvicorn import run

import torch


class Predictor(object):
    def __init__(self, model_mode = MODEL_MODE, device_mode = USE_GPU):
        self.model = _initialize_model(model_mode, device_mode)
    
    def predict(self, sequences):
        dataloader = load_from_array(sequences)
        with torch.no_grad():
            all_predicts = []
            for i, batch in enumerate(dataloader):
                best_paths = self.model.forward(batch)
                seq_lengths = batch["features"]["seq_lengths"]
                idxs_to_tags = lambda idxs: [IDX_TO_TAG[idx] for idx in idxs]
                for j in range(len(batch)):
                    sample_predicts = best_paths[j][: seq_lengths[j]]
                    all_predicts.append(idxs_to_tags(sample_predicts))
            return all_predicts

def _initialize_model(model_mode, device_mode):

    """
    Create NERModel Instance with:
        model_mode: type of model.
        device_mode: cpu(0) or gpu - cuda(1).
    """
    device_name = "cpu"
    device = torch.device(device_name)

    if device_mode == 1:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

    model_configs = get_configs(model_mode)

    model = NERModel(
        model_configs["lstm"],
        model_configs["char_level_embedding"],
        model_configs["casing_embedding"],
        model_configs["crf"],
        device,
    ).to(device)
    weights_path = ROOT / "modules" / "models" / model_configs["model_name"] / "best.w"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model

