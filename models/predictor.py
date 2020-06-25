"""
@author: Li Xi
@file: predictor.py
@time: 2019-05-11 22:55
@desc:
"""

import sys

sys.path.append('..')

from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.models.model import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('Event')
class EventPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, chain: list, event_tp: str) -> JsonDict:
        return self.predict_json({"chain": chain, "event_type": event_tp})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        chain = json_dict["chain"]
        event_type = json_dict["event_type"]
        return self._dataset_reader.text_to_instance(chain, event_type)
