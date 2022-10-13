import os
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
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import argparse
import torch
from pydantic import BaseModel

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


def _initialize_server(model):
    """
    Create a Server Instance with apis from NERModel.
    """
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Welcome to the Food Vision API!"}

    class Item(BaseModel):
        sequences: list

    @app.post("/predict/")
    async def predict(item: Item):
        dataloader = load_from_array(item.sequences)
        with torch.no_grad():
            all_predicts = []
            for i, batch in enumerate(dataloader):
                best_paths = model.forward(batch)
                seq_lengths = batch["features"]["seq_lengths"]
                idxs_to_tags = lambda idxs: [IDX_TO_TAG[idx] for idx in idxs]
                for j in range(len(batch)):
                    sample_predicts = best_paths[j][: seq_lengths[j]]
                    all_predicts.append(idxs_to_tags(sample_predicts))
            return all_predicts

    return app


def start(model_mode = MODEL_MODE, device_mode = USE_GPU, port = PORT):
    """
    Create server and run on a port.
    """
    model = _initialize_model(model_mode, device_mode)
    app = _initialize_server(model)
    port = int(os.environ.get("PORT", port))
    run(app, port=port)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-mode", type=int, default=MODEL_MODE, help="mode of model"
    )
    parser.add_argument(
        "--device-mode", type=int, default=USE_GPU, help="0: cpu - 1: cuda"
    )
    parser.add_argument("--port", type=int, default=PORT, help="localhost: port")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    start(opt.model_mode, opt.device_mode, opt.port)
