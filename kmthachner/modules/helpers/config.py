""" 
  Functions for initialize parameters of each layer in model,
  then save into dict object.
"""
def create_cnn_embedding_config(
    num_vocabs, vectorize_dim, in_channels, out_channels, kernel_size, padding_size
):
    config = {"mode": "CNN"}
    config["num_vocabs"] = num_vocabs
    config["vectorize_dim"] = vectorize_dim
    config["in_channels"] = in_channels
    config["out_channels"] = out_channels
    config["kernel_size"] = kernel_size
    config["padding_size"] = padding_size

    return config


def create_lstm_embedding_config(
    num_vocabs, vectorize_dim, hidden_size, num_layers, bidirectional
):
    config = {"mode": "LSTM"}
    config["num_vocabs"] = num_vocabs
    config["vectorize_dim"] = vectorize_dim
    config["hidden_size"] = hidden_size
    config["num_layers"] = num_layers
    config["bidirectional"] = bidirectional

    return config


def create_casing_config(num_vocabs, vectorize_dim):
    config = {}
    config["num_vocabs"] = num_vocabs
    config["vectorize_dim"] = vectorize_dim

    return config


def create_crf_config(target_size):
    config = {}
    config["target_size"] = target_size

    return config


def create_lstm_config(input_size, hidden_size, p_dropout):
    config = {}
    config["input_size"] = input_size
    config["hidden_size"] = hidden_size
    config["p_dropout"] = p_dropout

    return config
