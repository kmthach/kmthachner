import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    

from constants.model_constants import *
from defaults.model_defaults import *
from defaults.tag_dict import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(SEED)

class NERModel(nn.Module):
    def __init__(
        self,
        lstm_config,
        char_level_embedding_config,
        casing_embedding_config,
        crf_config,
        device,
    ):
        super(NERModel, self).__init__()
        self.device = device

        self.tag_to_idx = TAG_TO_IDX
        self.use_char_level_embedding = char_level_embedding_config != None
        self.use_casing_embedding = casing_embedding_config != None
        self.use_crf = crf_config != None

        # Initialize Layers base on config dict oject
        if self.use_char_level_embedding:
            self.embedding_mode = char_level_embedding_config["mode"]
            self.char_vectorize = nn.Embedding(
                num_embeddings=char_level_embedding_config["num_vocabs"],
                embedding_dim=char_level_embedding_config["vectorize_dim"],
            )

            if self.embedding_mode == "CNN":
                cnn_embedding_config = char_level_embedding_config

                self.char_level_embeds = nn.Conv2d(
                    in_channels=cnn_embedding_config["in_channels"],
                    out_channels=cnn_embedding_config["out_channels"],
                    kernel_size=cnn_embedding_config["kernel_size"],
                    padding=cnn_embedding_config["padding_size"],
                )

            elif self.embedding_mode == "LSTM":
                lstm_embedding_config = char_level_embedding_config
                self.char_level_embeds = nn.LSTM(
                    input_size=lstm_embedding_config["vectorize_dim"],
                    hidden_size=lstm_embedding_config["hidden_size"],
                    num_layers=lstm_embedding_config["num_layers"],
                    bidirectional=lstm_embedding_config["bidirectional"],
                    batch_first=True,
                )
        if self.use_casing_embedding:
            self.casing_embeds = nn.Embedding(
                num_embeddings=casing_embedding_config["num_vocabs"],
                embedding_dim=casing_embedding_config["vectorize_dim"],
            )

        self.dropout = nn.Dropout(lstm_config["p_dropout"])
        self.lstm = nn.LSTM(
            input_size=lstm_config["input_size"],
            hidden_size=lstm_config["hidden_size"],
            bidirectional=True,
            batch_first=True,
        )

        target_size = len(self.tag_to_idx)
        self.hidden2tag = nn.Linear(lstm_config["hidden_size"] * 2, target_size)

        if self.use_crf:
            self.tag_to_idx[START] = START_IDX
            self.tag_to_idx[STOP] = STOP_IDX

            target_size = crf_config["target_size"]

            self.transitions = nn.Parameter(torch.randn(target_size, target_size))
            self.transitions.data[:, self.tag_to_idx[START]] = IMPOSSIBLE
            self.transitions.data[self.tag_to_idx[STOP]] = IMPOSSIBLE

            self.hidden2tag = nn.Linear(lstm_config["hidden_size"] * 2, target_size)

    def _get_lstm_features(self, batch):
        """
        Fit batch and return the outputs of lstm layers.
        """
        embeds = batch["global_vectors"].to(self.device)
        if self.use_char_level_embedding:

            char_vectorize_outputs = self.char_vectorize(
                batch["char_level_vectors"].to(torch.int64).to(self.device)
            )

            if self.embedding_mode == "CNN":
                char_embeds = self.char_level_embeds(
                    char_vectorize_outputs.view(
                        -1,
                        1,
                        char_vectorize_outputs.shape[2],
                        char_vectorize_outputs.shape[3],
                    )
                )

                char_embeds = char_embeds.view(
                    char_vectorize_outputs.shape[0],
                    char_vectorize_outputs.shape[1],
                    char_embeds.shape[2],
                    -1,
                )

                char_embeds = nn.functional.max_pool2d(
                    char_embeds, kernel_size=(char_vectorize_outputs.size(2), 1)
                ).squeeze(dim=2)

                embeds = torch.cat([embeds, char_embeds], dim=2)

            elif self.embedding_mode == "LSTM":
                out, (char_embeds, cn) = self.char_level_embeds(
                    char_vectorize_outputs.view(
                        -1,
                        char_vectorize_outputs.shape[2],
                        char_vectorize_outputs.shape[3],
                    )
                )
                char_embeds = char_embeds.view(
                    char_vectorize_outputs.shape[0], char_vectorize_outputs.shape[1], -1
                )
                embeds = torch.cat([embeds, char_embeds], dim=2)

        if self.use_casing_embedding:
            casing_embeds = self.casing_embeds(
                batch["casing_vectors"].to(torch.int64).to(self.device)
            )

            embeds = torch.cat([embeds, casing_embeds], dim=2)

        embeds = self.dropout(embeds)

        packed_embeds = pack_padded_sequence(
            embeds, batch["seq_lengths"], batch_first=True
        )
        lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def fit_and_compute_loss(self, batch):
        """
        Fit batch to model then return loss.
        """
        batch_lstm_feats = self._get_lstm_features(batch["features"])

        if self.use_crf:
            loss = self.crf_loss(
                batch_lstm_feats, batch["tags"], batch["features"]["seq_lengths"]
            )
            return loss
        else:
            loss = self.cross_entropy_loss(
                batch_lstm_feats,
                batch["tags"],
                torch.Tensor(batch["features"]["seq_lengths"]).to(self.device),
            )
            return loss

    def forward(self, batch):
        # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        batch_lstm_feats = self._get_lstm_features(batch["features"])
        if self.use_crf:
            max_sequence_len = batch["features"]["seq_lengths"][0]
            batch_size = len(batch["features"]["seq_lengths"])
            masks = torch.zeros(batch_size, max_sequence_len)
            for i, length in enumerate(batch["features"]["seq_lengths"]):
                masks[i, :length] = 1
            # Find the best path, given the features.
            score, tag_seq = self.__viterbi_decode(
                batch_lstm_feats,
                masks[:, : batch_lstm_feats.size(1)].float().to(self.device),
            )

            return tag_seq

        else:
            output = torch.argmax(batch_lstm_feats, dim=2).tolist()
            return output

    def cross_entropy_loss(self, batch_lstm_feats, tags, seq_lengths):
        """
        Calculate average cross entropy loss of batch by masking
        the padding words.
        """
        flatten_output = batch_lstm_feats.view(-1, batch_lstm_feats.shape[2])
        flatten_true_tags = tags.view(-1, 1).long().to(self.device)

        log_probs_flat = nn.functional.log_softmax(flatten_output, dim=1)

        losses_flat = -torch.gather(log_probs_flat, dim=1, index=flatten_true_tags)

        bs = batch_lstm_feats.shape[0]
        max_len = batch_lstm_feats.shape[1]
        losses = losses_flat.view(bs, max_len)

        seq_range = torch.arange(0, max_len).long().expand(bs, max_len).to(self.device)
        expanded_len = seq_lengths.view(bs, 1).expand(bs, max_len)
        mask = (expanded_len > seq_range).float()

        losses = losses * mask.float()
        loss = losses.sum() / seq_lengths.float().sum()
        return loss

    def log_sum_exp(self, x):
        """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))"""
        max_score = x.max(-1)[0]
        return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()

    def crf_loss(self, features, tags, seq_lengths):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        masks = torch.ones_like(tags).to(self.device)
        for i, length in enumerate(seq_lengths):
            masks[i, :length] = 1
        L = features.size(1)
        masks_ = masks[:, :L]

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(
            features, tags[:, :L].long().to(self.device), masks_
        )
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full(
            (B, 1), self.tag_to_idx[START], dtype=torch.long, device=tags.device
        )
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(
            1
        )  # [B]
        last_score = self.transitions[self.tag_to_idx[STOP], last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(
            B, L, C, dtype=torch.long, device=features.device
        )  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.tag_to_idx[STOP]] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (
                1 - mask_t
            )  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.tag_to_idx[STOP]]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.tag_to_idx[START]] = 0.0
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = (
                scores.unsqueeze(1) + trans + emit_score_t
            )  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = self.log_sum_exp(score_t)  # [B, C
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = self.log_sum_exp(scores + self.transitions[self.tag_to_idx[STOP]])
        return scores
