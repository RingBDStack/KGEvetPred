"""
@author: Li Xi
@file: train.py
@time: 2020/2/7 21:09
@desc:
"""

import argparse
import json
import os
import time

import gensim
import numpy as np
import torch
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from torch import optim

from models.AttentionLSTM import AttentionLSTM
from models.MemNet import MemNet
from models.GRMN import GRMN
from models.ATAE import ATAE
from models.EventLSTM import EventLSTM
from models.GMN import GMN
from models.event_reader import EventDataReader
from models.predictor import EventPredictor
from my_logger import logger

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['CUDA_CACHE_PATH'] = '/home/LAB/lixi/cuda_cache'


if __name__ == "__main__":

    # option dict

    model_names = dict()
    model_names[0] = 'GRMN'
    model_names[1] = 'GMN'
    model_names[2] = 'MemNet'
    model_names[3] = 'ATAE'
    model_names[4] = 'EvtLSTM'
    model_names[5] = 'EvtAttLSTM'
    model_names[6] = 'DA-EvtLSTM'
    model_names[7] = 'DA-EvtAttLSTM'

    event_types = dict()
    event_types[-1] = ''
    event_types[0] = '爆炸'
    event_types[1] = '火灾'
    event_types[2] = '地质 灾害'
    event_types[3] = '交通 事故'
    event_types[4] = '人身 伤害'

    # run config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m', required=True, type=int)
    parser.add_argument("--embedding_size", '-emb', required=True, type=int)
    parser.add_argument("--learning_rate", '-lr', required=True, type=float)
    parser.add_argument("--batch_size", '-bs', required=True, type=int)
    parser.add_argument("--patience", '-pc', required=True, type=int)
    parser.add_argument("--epochs", '-ep', required=True, type=int)
    parser.add_argument("--hidden_size", '-hs', required=True, type=int)
    parser.add_argument("--hop_num", '-hn', required=True, type=int)
    parser.add_argument("--file_dir", '-fd', required=True, type=int)
    parser.add_argument("--threshold", '-th', required=True, type=float)
    parser.add_argument("--event_type", '-et', required=True, type=int)
    # python train.py -m 0 -emb 300 -lr 1e-4 -bs 5 -pc 1 -ep 10 -hs 16 -hn 6 -fd 1 -th 0.5 -et 0

    args = parser.parse_args()
    args.cuda_device = [0]  # [0, 1, 2, 3]
    args.label_num = 2
    args.threshold = 0.5
    model_name = model_names[args.model]
    torch.backends.cudnn.enabled = False

    logger.debug('------------------------')
    logger.debug('------------------------')
    logger.debug('-------Parameters-------')
    logger.debug('{}: {}'.format('model', model_name))
    logger.debug(args)

    # load data
    # TODO: file path
    reader = EventDataReader()
    train_dataset = ensure_list(reader.read('./dataset/1_{}/{}/train.data'.format(args.file_dir, args.event_type)))
    eval_dataset = ensure_list(reader.read('./dataset/1_{}/{}/eval.data'.format(args.file_dir, args.event_type)))
    with open('./dataset/1_{}/{}/test.data'.format(args.file_dir, args.event_type), 'r', encoding='utf-8') as f:
        test_dataset = json.loads(f.read())

    # reader = EventDataReader()
    # train_dataset = ensure_list(reader.read(os.path.join('.', 'dataset', 'example', 'train.data')))
    # eval_dataset = ensure_list(reader.read(os.path.join('.', 'dataset', 'example', 'train.data')))
    # with open(os.path.join('.', 'dataset', 'example', 'train.data'), 'r', encoding='utf-8') as f:
    #     test_dataset = json.loads(f.read())

    # get vocabulary and embedding
    vocab = Vocabulary.from_instances(train_dataset + eval_dataset,
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
                                                 "object_attri_4": 0,
                                                 "event_type": 0})

    # load pre-trained word vector
    word_vector_path = os.path.join('.', 'dataset', 'sgns.event')

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

    model = None
    if model_name == 'GRMN':
        model = GRMN(args, word_embeddings, vocab)
    elif model_name == 'GMN':
        model = GMN(args, word_embeddings, vocab)
    elif model_name == 'MemNet':
        model = MemNet(args, word_embeddings, vocab)
    elif model_name == 'ATAE':
        model = ATAE(args, word_embeddings, vocab)
    elif model_name == 'EvtLSTM':
        model = EventLSTM(args, word_embeddings, vocab, domain_info=False)
    elif model_name == 'EvtAttLSTM':
        model = AttentionLSTM(args, word_embeddings, vocab, domain_info=False)
    elif model_name == 'DA-EvtLSTM':
        model = EventLSTM(args, word_embeddings, vocab, domain_info=True)
    elif model_name == 'DA-EvtAttLSTM':
        model = AttentionLSTM(args, word_embeddings, vocab, domain_info=True)

    if args.cuda_device != -1:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("trigger_0", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=eval_dataset,
                      shuffle=False,  # ensure label 0 has a label id 0
                      num_epochs=args.epochs,
                      patience=args.patience,  # stop training before loss raise
                      cuda_device=args.cuda_device,  # cuda device id
                      # serialization_dir=os.path.join('.', 'tensorboard'),
                      # summary_interval=2,
                      # histogram_interval=2,
                      # should_log_parameter_statistics=True,
                      # should_log_learning_rate=True
                      )

    # start train
    metrics = trainer.train()

    # save model
    model_path = './checkpoints/{}_fd_{}_bs_{}__et_{}_{}.tar.gz'.format(model_name, args.file_dir, args.batch_size, args.event_type, str(
        time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))))
    torch.save(model.state_dict(), model_path)
    logger.debug('save model in {}'.format(model_path))

    # start predict
    predictor = EventPredictor(model, dataset_reader=reader)

    predict_out = []
    predict_out_f1 = []
    label_y = []

    for event_item in test_dataset:
        event = event_item['event']
        label = event_item['label']
        event_type = event_item['event_type']
        logits = predictor.predict(event, event_type)['logits']
        predict_out.append(logits)
        label_y.append(label)
        if logits == 1:
            predict_out_f1.append([0, 1])
        else:
            predict_out_f1.append([1, 0])

    predict_out = torch.LongTensor(predict_out)
    predict_out_f1 = torch.LongTensor(predict_out_f1)
    label_y = torch.LongTensor(label_y)

    # metrics
    get_accuracy = BooleanAccuracy()
    get_f1_score = F1Measure(positive_label=1)

    get_accuracy(predict_out, label_y)
    accuracy = get_accuracy.get_metric(reset=False)
    get_f1_score(predict_out_f1, label_y)
    precision, recall, f1_measure = get_f1_score.get_metric(reset=False)

    logger.debug('-------Train Metrics-------')
    for k in metrics:
        logger.debug('{}: {}'.format(k, metrics[k]))
    logger.debug('-------Test Output-------')
    logger.debug('accuracy: {}'.format(accuracy))
    logger.debug('precision: {}'.format(precision))
    logger.debug('recall: {}'.format(recall))
    logger.debug('f1_measure: {}'.format(f1_measure))
