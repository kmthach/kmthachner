import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from defaults.featuring_defaults import *

import torch
import numpy as np

class Featuring(object):
    """
    Use for transfrom raw data to features
    """

    def __init__(self, use_char_level, use_casing):
        self.use_char_level = use_char_level
        self.use_casing = use_casing
        embeddings_dict = {}

        # Read pretrain data and save into embeddings_dict
        with open(ROOT / PRETRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor(np.asarray(values[1:], "float32"))
                embeddings_dict[word] = vector
        self.embeddings_dict = embeddings_dict

    def tokens_to_glove_vectors(self, tokens):
        # Convert tokens to GloVe
        word_to_vec = lambda word: self.embeddings_dict.get(
            word.lower(), torch.zeros(GLOVE_DIM)
        )

        global_vectors = torch.stack([word_to_vec(token) for token in tokens])
        return global_vectors

    def tokens_to_char_level_vectors(self, tokens):
        all_characters = ALL_CHARACTERS
        char_dict = {c: idx + 1 for idx, c in enumerate(all_characters)}

        token_to_seq = lambda token: [char_dict[char] for char in token]

        char_level_vectors = [token_to_seq(token) for token in tokens]
        return char_level_vectors

    def tokens_to_casing_vectors(self, tokens):
        def token_to_casing(token):
            casing = 1

            numDigits = 0
            for char in token:
                if char.isdigit():
                    numDigits += 1

            digitFraction = numDigits / float(len(token))

            if token.isdigit():  # Is a digit
                casing = 2
            elif digitFraction > 0.5:
                casing = 3
            elif token.islower():  # All lower case
                casing = 4
            elif token.isupper():  # All upper case
                casing = 5
            elif token[0].isupper():  # is a title, initial char upper, then all lower
                casing = 6
            elif numDigits > 0:
                casing = 7

            return casing

        casing_vectors = [token_to_casing(token) for token in tokens]
        return casing_vectors

    def preprocess(self, tokens, tags):
        converted_sample = {"features": {}}

        if tags != None:
            converted_sample["tags"] = tags
        global_vectors = self.tokens_to_glove_vectors(tokens)
        converted_sample["features"]["global_vectors"] = global_vectors
        if self.use_char_level:

            char_level_vectors = self.tokens_to_char_level_vectors(tokens)
            converted_sample["features"]["char_level_vectors"] = char_level_vectors

        if self.use_casing:
            casing_vectors = self.tokens_to_casing_vectors(tokens)
            converted_sample["features"]["casing_vectors"] = casing_vectors
        return converted_sample
