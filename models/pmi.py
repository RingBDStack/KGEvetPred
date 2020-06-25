from __future__ import division

import json
import random

# TODO: change dir path and file path
test_dir = r'../corpus/bad.data'
train_file = r'../corpus/good.data'


class Event:
    def __init__(self, verb, sbj, obj):
        self.verb = verb
        self.sbj = sbj
        self.obj = obj
        self.iobj = ''


class Question:
    def __init__(self, answer, context, choices):
        self.answer = answer
        self.context = context
        self.choices = choices


class PMI:
    def __init__(self, train_questions):
        self.V = {}
        self.N = 0
        for q in train_questions:
            verb_list = [event.verb for event in q.context]
            verb_list.append(q.choices[q.answer].verb)
            self.N += len(verb_list)

            for i in range(len(verb_list)):
                for j in range(i + 1, len(verb_list)):
                    verb_a, verb_b = verb_list[i], verb_list[j]
                    self.add_to_v(verb_a, verb_b)
                    self.add_to_v(verb_b, verb_a)

    def add_to_v(self, verb_a, verb_b):
        if verb_a not in self.V:
            self.V[verb_a] = {}
        if verb_b not in self.V[verb_a]:
            self.V[verb_a][verb_b] = 0
        self.V[verb_a][verb_b] += 1

    def get_pmi_score(self, verb_a, verb_b):
        if verb_a in self.V and verb_b in self.V and verb_b in self.V[verb_a]:
            p_a_b = self.V[verb_a][verb_b]
            p_a = sum(self.V[verb_a].values())
            p_b = sum(self.V[verb_b].values())
            # return math.log(self.N*p_a_b/(p_a*p_b))
            return self.N * p_a_b / (p_a * p_b)
        else:
            return 0

    def get_most_similar_choice(self, test_question):
        results = []
        for i in range(len(test_question.choices)):
            similarity = sum(
                [self.get_pmi_score(context_event.verb, test_question.choices[i].verb) for context_event in
                 test_question.context])
            results.append((similarity, i))
        results.sort()
        return results[-1][1]


def pmi_prediction(train, test):
    pmi = PMI(train)
    results = []
    for test_question in test:
        choice_index = pmi.get_most_similar_choice(test_question)
        results.append(choice_index)
    return results


def parse_event(eventdic):
    verb = eventdic['trigger']
    sbj = eventdic['trigger_agent']
    obj = eventdic['trigger_object']
    event = Event(verb, sbj, obj)
    return event


def read_question_corpus(path):
    items = []
    with open(path, 'r') as load_f:
        data_list = json.load(load_f)
    for chain in data_list:
        answer = 0
        context = [parse_event(event) for event in chain]
        choices = [parse_event(event) for event in chain]
        items.append(Question(answer, context, choices))
    return items


def build_question(chains, all_verbs):
    answer = 0
    context = chains[:-1]
    choices = []
    choices.append(chains[-1])
    for v in all_verbs:
        if len(choices) < 5:
            if v != chains[-1].verb:
                choices.append(Event(v, '', ''))
        else:
            break
    return Question(answer, context, choices)


def read_c_and_j_corpus():
    documents = []
    all_verbs = []
    with open(test_dir, 'r') as load_f:
        data_list = json.load(load_f)
    for data in data_list:
        chain = []
        for event in data:
            chain.append(Event(event['trigger'], '', ''))
            all_verbs.append(event['trigger'])
        documents.append(chain)
    all_verbs = [v for v in set(all_verbs)]
    questions = []
    for chains in documents:
        if len(chains) < 9:
            random.shuffle(all_verbs)
            questions.append(build_question(chains, all_verbs))
        else:
            for i in range(0, len(chains) - 9):
                random.shuffle(all_verbs)
                questions.append(build_question(chains[i:i + 9], all_verbs))

    return questions


def eval(test, results):
    acc = 0
    for i in range(len(test)):
        if results[i] == test[i].answer:
            acc += 1
    print('Acc:%s' % (acc / len(test)))


if __name__ == '__main__':
    train = read_question_corpus(train_file)
    test = read_c_and_j_corpus()
    train = train[:10000]
    results = pmi_prediction(train, test)
    eval(test, results)
