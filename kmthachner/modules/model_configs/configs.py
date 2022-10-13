import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from defaults.featuring_defaults import GLOVE_DIM
from defaults.model_defaults import *
from helpers.config import *

import json

def get_configs(mode):

    """
    Read Supported Models in configs.json then create a dict
    to save chosen model settings.
    """
    lstm_input_size = GLOVE_DIM
    f = open(ROOT / "model_configs/configs.json")
    model_dict = json.load(f)
    model_config = list(model_dict.values())[mode]

    model_config["model_name"] = list(model_dict.keys())[mode]
    model_config["char_level_embedding"] = None
    model_config["casing_embedding"] = None
    model_config["crf"] = None
    if model_config["use_char_level_embedding"]:

        if model_config["use_char_level_embedding"] == "CNN":
            model_config["char_level_embedding"] = create_cnn_embedding_config(
                CHAR_EMBEDDING_NUM_VOCABS,
                CHAR_EMBEDDING_VECTORIZE_DIM,
                CNN_EMBEDDING_IN_CHANNELS,
                CNN_EMBEDDING_OUT_CHANNELS,
                CNN_EMBEDDING_KERNEL_SIZE,
                CNN_EMBEDDING_PADDING_SIZE,
            )
            lstm_input_size += CNN_EMBEDDING_OUT_CHANNELS

        elif model_config["use_char_level_embedding"] == "LSTM":
            model_config["char_level_embedding"] = create_lstm_embedding_config(
                CHAR_EMBEDDING_NUM_VOCABS,
                CHAR_EMBEDDING_VECTORIZE_DIM,
                LSTM_EMBEDDING_HIDDEN_SIZE,
                LSTM_EMBEDDING_NUM_LAYERS,
                LSTM_EMBEDDING_BIDIRECTIONAL,
            )
            lstm_input_size += 2 * LSTM_EMBEDDING_HIDDEN_SIZE

    if model_config["use_casing_feature"]:
        model_config["casing_embedding"] = create_casing_config(
            CASING_NUM_VOCABS, CASING_VECTORIZE_DIM
        )
        lstm_input_size += CASING_VECTORIZE_DIM

    if model_config["use_crf"]:
        model_config["crf"] = create_crf_config(CRF_TARGET_SIZE)

    model_config["lstm"] = create_lstm_config(
        lstm_input_size, LSTM_HIDDEN_SIZE, P_DROPOUT
    )

    return model_config
