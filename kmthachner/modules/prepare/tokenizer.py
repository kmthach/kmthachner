from torchtext.data.utils import get_tokenizer

def tokenize(corpus):
    # tokenize sequence
    tokenizer = get_tokenizer("spacy")
    tokens_sequences = []
    for sentence in corpus:
        tokens_sequences.append(tokenizer(sentence))

    return tokens_sequences
