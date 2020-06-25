"""
@author: Li Xi
@file: event_reader.py
@time: 2019-04-28 13:58
@desc:
"""

import json
import os
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader, TokenIndexer, Instance, Tokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides


@DatasetReader.register("event_data")
class EventDataReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or JustSpacesWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding='utf-8') as data_file:
            chains = json.loads(data_file.read())
            for chain in chains:
                event = chain['event']
                label = chain['label']
                event_type = chain['event_type']
                yield self.text_to_instance(event, event_type, label)

    @overrides
    def text_to_instance(self, chain: list, event_type: str, label: int = None) -> Instance:
        keys = [x for x in chain[0].keys() if x != 'id']

        fields = {}
        for i in range(len(chain)):
            for k in keys:
                tokenized_info = self._tokenizer.split_words(chain[i][k])
                info_field = TextField(tokenized_info, self._token_indexers)
                fields[k + '_' + str(i)] = info_field

        tokenized_type = self._tokenizer.split_words(event_type)
        type_field = TextField(tokenized_type, self._token_indexers)
        fields['event_type'] = type_field

        if label is not None:
            fields['label'] = LabelField(str(label))

        return Instance(fields)


if __name__ == '__main__':
    reader = EventDataReader()
    instances = ensure_list(reader.read(os.path.join('..', 'dataset', 'example', 'train.data')))

    for instance in instances:
        fields = instance.fields

        for k in fields:
            if k != 'label':
                print(k, [w.text for w in fields[k].tokens])
                print('--------------')
            elif k == 'label':
                print('label', fields[k].label)
