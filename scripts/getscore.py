"""
@author: Li Xi
@file: getscore.py
@time: 2020/6/12 15:33
@desc:
"""
import argparse
import json
import os

import gensim
import numpy as np
import torch
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


from models.AttentionLSTM import AttentionLSTM
from models.MemNet import MemNet
from models.GRMN import GRMN
from models.ATAE import ATAE
from models.EventLSTM import EventLSTM
from models.GMN import GMN

from models.event_reader import EventDataReader
from models.predictor import EventPredictor

import json
for i in range(-1, 5, 1):
    with open('./dataset/1_4/{}/train.data'.format(str(i)),'r', encoding='utf-8') as f:
        try:
            content = json.loads(f.read())
            print(len(content))
        except Exception as e:
            print(i)
exit(0)
# 参数

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
args = parser.parse_args()

models = ["DA-EvtLSTM"  ,"EvtLSTM","MemNet" ,"GMN",  "ATAE" ,"GRMN"  ]

args.batch_size = 32
model_name =models[0]


event_type = [-1]

args.embedding_size = 300
args.learning_rate = 1e-3
args.cuda_device = [0]
args.hidden_size = 32
args.hop_num = 3
args.label_num = 2
args.threshold = 0.5

# 读取文件

for et in event_type:

    reader = EventDataReader()
    train_dataset = ensure_list(reader.read(os.path.join('.', 'dataset', '1_4', str(et), 'train.data')))
    eval_dataset = ensure_list(reader.read(os.path.join('.', 'dataset', '1_4', str(et), 'eval.data')))
    with open(os.path.join('.', 'dataset', '1_4', str(et), 'test.data'), 'r', encoding='utf-8') as f:
        test_dataset = json.loads(f.read())
    print('load done')


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
    word_vector_path = os.path.join('.', 'dataset', 'sgns.event')
    word_vector = gensim.models.KeyedVectors.load_word2vec_format(word_vector_path)
    pretrained_weight = np.array([[0.00] * args.embedding_size] * vocab.get_vocab_size())

    for i in range(vocab.get_vocab_size()):
        word = vocab.get_token_from_index(i, 'tokens')
        if word in word_vector.vocab:
            pretrained_weight[vocab.get_token_index(word)] = word_vector[word]

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=args.embedding_size, weight=torch.from_numpy(pretrained_weight).float())
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    print('prepare done')




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




    checkpoints_path = "checkpoints"
    checkpoints = os.listdir(checkpoints_path)

    checkpoints = [x
                   for x in checkpoints if
                   model_name in x and
                   "bs_" + str(args.batch_size) in x and
                   "et_" + str(et) in x
                   ]




    for ckpt in checkpoints:
        state_path = os.path.join(checkpoints_path, ckpt)
        print('state_path:', state_path)
        model.load_state_dict(torch.load(state_path))

        predictor = EventPredictor(model, dataset_reader=reader)

        predict_out = []
        predict_out_f1 = []
        label_y = []
        scores = []

        for event_item in test_dataset:
            event = event_item['event']
            label = event_item['label']
            event_tp = event_item["event_type"]

            logits = predictor.predict(event, event_tp)['logits']
            predict_out.append(logits)
            score = predictor.predict(event, event_tp)['score'][0]
            scores.append(score)
            label_y.append(label)
            if logits == 1:
                predict_out_f1.append([0, 1])
            else:
                predict_out_f1.append([1, 0])


        with open("output/"+ckpt+'.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps({"scores":scores, "label_y": label_y},
                       ensure_ascii=False))

        print("done!!! model {} batchsize {} event type {}".format(model_name, str(args.batch_size), str(et)))
