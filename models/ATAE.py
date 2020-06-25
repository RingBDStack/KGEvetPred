"""
@author: Li Xi
@file: ATAE.py
@time: 2019-05-11 20:44
@desc:
"""
import argparse
import os
from typing import Dict

import gensim
import numpy as np
import torch
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from overrides import overrides
from torch import optim
from torch.nn import BCELoss

from models.event_reader import EventDataReader
from models.layers.attention import NoQueryAttention
from models.layers.decoder import Score
from models.layers.dynamic_rnn import DynamicLSTM
from models.layers.encoder import EventEmbedding


@Model.register("ATAE")
class ATAE(Model):

    def __init__(self,
                 args,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        # parameters
        self.args = args
        self.word_embeddings = word_embeddings

        # layers
        self.event_embedding = EventEmbedding(self.args, self.word_embeddings)
        self.lstm = DynamicLSTM(self.args.embedding_size * 2, self.args.hidden_size, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(self.args.hidden_size + self.args.embedding_size * 2, score_function='bi_linear')
        self.score = Score(self.args.hidden_size, self.args.embedding_size, threshold=self.args.threshold)

        # metrics
        self.accuracy = BooleanAccuracy()
        self.f1_score = F1Measure(positive_label=1)
        self.loss_function = BCELoss()

    @overrides
    def forward(self,
                trigger_0: Dict[str, torch.LongTensor],
                trigger_agent_0: Dict[str, torch.LongTensor],
                agent_attri_0: Dict[str, torch.LongTensor],
                trigger_object_0: Dict[str, torch.LongTensor],
                object_attri_0: Dict[str, torch.LongTensor],
                trigger_1: Dict[str, torch.LongTensor],
                trigger_agent_1: Dict[str, torch.LongTensor],
                agent_attri_1: Dict[str, torch.LongTensor],
                trigger_object_1: Dict[str, torch.LongTensor],
                object_attri_1: Dict[str, torch.LongTensor],
                trigger_2: Dict[str, torch.LongTensor],
                trigger_agent_2: Dict[str, torch.LongTensor],
                agent_attri_2: Dict[str, torch.LongTensor],
                trigger_object_2: Dict[str, torch.LongTensor],
                object_attri_2: Dict[str, torch.LongTensor],
                trigger_3: Dict[str, torch.LongTensor],
                trigger_agent_3: Dict[str, torch.LongTensor],
                agent_attri_3: Dict[str, torch.LongTensor],
                trigger_object_3: Dict[str, torch.LongTensor],
                object_attri_3: Dict[str, torch.LongTensor],
                trigger_4: Dict[str, torch.LongTensor],
                trigger_agent_4: Dict[str, torch.LongTensor],
                agent_attri_4: Dict[str, torch.LongTensor],
                trigger_object_4: Dict[str, torch.LongTensor],
                object_attri_4: Dict[str, torch.LongTensor],
                event_type: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # tri, e: [batch_size, 1, embedding_size]
        tri0, e0 = self.event_embedding(trigger_0, trigger_agent_0, trigger_object_0)
        tri1, e1 = self.event_embedding(trigger_1, trigger_agent_1, trigger_object_1)
        tri2, e2 = self.event_embedding(trigger_2, trigger_agent_2, trigger_object_2)
        tri3, e3 = self.event_embedding(trigger_3, trigger_agent_3, trigger_object_3)
        tri4, e4 = self.event_embedding(trigger_4, trigger_agent_4, trigger_object_4)

        # [batch_size, seq_len, embedding_size]
        e = (torch.stack([e0, e1, e2, e3, e4], dim=1)).squeeze(2)
        # [batch_size, seq_len, embedding_size]
        x = (torch.stack([tri4, tri4, tri4, tri4, tri4], dim=1)).squeeze(2)
        # [batch_size, seq_len, embedding_size * 2]
        x = torch.cat((e, x), dim=-1)
        batch_size = x.size()[0]
        # [batch_size, seq_len, hidden_size]
        h, (_, _) = self.lstm(x, torch.LongTensor([5] * batch_size))
        # [batch_size, seq_len, hidden_size + embedding_size * 2]
        ha = torch.cat((h, x), dim=-1)
        # [batch_size, 1, seq_len]
        _, score = self.attention(ha)
        # [batch_size, hidden_size]
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        # [batch_size, 1] , [batch_size], [batch_size, label_size]
        score, logits, logits_f1 = self.score(output, tri4)

        output = {"logits": logits,
                  "score": score}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_score(logits_f1, label)
            output["loss"] = self.loss_function(score.squeeze(1), label.float())

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.accuracy.get_metric(reset)
        precision, recall, f1_measure = self.f1_score.get_metric(reset)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_measure": f1_measure
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.embedding_size = 300
    args.learning_rate = 1e-4
    args.batch_size = 1
    args.epochs = 10
    args.patience = 1
    args.cuda_device = -1
    args.hidden_size = 8
    args.hop_num = 6
    args.label_num = 2
    args.threshold = 0.5

    # load data
    reader = EventDataReader()
    train_dataset = ensure_list(reader.read(os.path.join('..','dataset','example','train.data')))
    eval_dataset = ensure_list(reader.read(os.path.join('..','dataset','example','train.data')))
    test_dataset = ensure_list(reader.read(os.path.join('..','dataset','example','train.data')))

    # get vocabulary and embedding
    vocab = Vocabulary.from_instances(train_dataset + eval_dataset + test_dataset,
                                      min_count={"trigger_0": 0,
                                                 "trigger_agent_0": 0,
                                                 "agent_attri_0": 0,
                                                 "trigger_object_0": 0,
                                                 "object_attri_0": 0,
                                                 "trigger_1": 0,
                                                 "trigger_agent_1": 0,
                                                 "agent_attri_1": 0,
                                                 "trigger_object_1": 0,
                                                 "object_attri_1": 0,
                                                 "trigger_2": 0,
                                                 "trigger_agent_2": 0,
                                                 "agent_attri_2": 0,
                                                 "trigger_object_2": 0,
                                                 "object_attri_2": 0,
                                                 "trigger_3": 0,
                                                 "trigger_agent_3": 0,
                                                 "agent_attri_3": 0,
                                                 "trigger_object_3": 0,
                                                 "object_attri_3": 0,
                                                 "trigger_4": 0,
                                                 "trigger_agent_4": 0,
                                                 "agent_attri_4": 0,
                                                 "trigger_object_4": 0,
                                                 "object_attri_4": 0})

    # load pre-trained word vector
    word_vector_path = os.path.join('..', 'dataset', 'sgns.event')
    word_vector = gensim.models.KeyedVectors.load_word2vec_format(word_vector_path)
    pretrained_weight = np.array([[0.00] * args.embedding_size] * vocab.get_vocab_size())

    for i in range(vocab.get_vocab_size()):
        word = vocab.get_token_from_index(i, 'tokens')
        if word in word_vector.vocab:
            pretrained_weight[vocab.get_token_index(word)] = word_vector[word]
    del word_vector

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=args.embedding_size, weight=torch.from_numpy(pretrained_weight).float())
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    model = ATAE(args, word_embeddings, vocab)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("trigger_0", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=eval_dataset,
                      num_epochs=args.epochs,
                      patience=args.patience,  # stop training before loss raise
                      cuda_device=args.cuda_device,  # cuda device id
                      )

    # start train
    metrics = trainer.train()
