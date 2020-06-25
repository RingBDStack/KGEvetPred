"""
@author: Li Xi
@file: formatter.py
@time: 2020/1/13 23:46
@desc:
"""
import os

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller


# 这里出现import error 不影响程序的正常运行

# ltp formatter 使用方法：
# 1）初始化LtpFormatter（如果有需要，则初始化ltp model的存储位置）
# 2）调用format方法，该方法输入是一个sentence，输出时dict格式数据
# 3）最后记得release
# 注意：format_only_pos是format的简化方法，这里只包含分词信息和词性标注信息
#
#
# 注：format输出格式：
# {
# 	'basic':[
# 		{
# 			'index':
# 			'word':
# 			'pos':
# 			'entity':
# 			'head':   arc.head 表示依存弧的父节点词的索引，ROOT节点的索引是-1，第一个词开始的索引依次为0、1、2、3 （原始root是0，index从1开始）
# 			'relation':  arc.relation 表示依存弧的关系，如：COO，ATT，VOB等
# 		},
# 		... ...
# 	],
# 	'role':[
# 		{
# 			'trigger':
#           'index':
# 			'relation':[
# 				{
# 					'name':
# 					'start'  与head不同，第一个词开始的索引依次为0、1、2…
# 					'end':
# 				},
# 				... ...
# 			]
# 		},
# 		... ...
# 	]
# }


class LtpFormatter:
    model_dir = os.path.join("utils", "ltp_data_v3.4.0")
    # 注意这里的位置需要调整为运行位置到ltp的相对位置，或者设置为绝对位置

    segmentor = Segmentor()
    segmentor.load(os.path.join(model_dir, "cws.model"))

    postagger = Postagger()
    postagger.load(os.path.join(model_dir, "pos.model"))

    parser = Parser()
    parser.load(os.path.join(model_dir, "parser.model"))

    recognizer = NamedEntityRecognizer()
    recognizer.load(os.path.join(model_dir, "ner.model"))

    labeller = SementicRoleLabeller()
    labeller.load(os.path.join(model_dir, "pisrl.model"))

    def format_only_pos(self, sentence):
        results = {'basic': [], 'role': []}

        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)

        index = 0
        for word, postag in zip(words, postags):
            results['basic'].append({
                'index': index,
                'word': word,
                'pos': postag
            })
            index += 1

        return results

    def format(self, sentence):

        results = {'basic': [], 'role': []}

        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)
        arcs = self.parser.parse(words, postags)
        netags = self.recognizer.recognize(words, postags)
        roles = self.labeller.label(words, postags, arcs)

        index = 0
        for word, postag, arc, netag in zip(words, postags, arcs, netags):
            results['basic'].append({
                'index': index,
                'word': word,
                'pos': postag,
                'entity': netag,
                'head': arc.head - 1,
                'relation': arc.relation
            })
            index += 1

        for role in roles:
            relations = []

            for arg in role.arguments:
                relations.append({
                    'name': arg.name,
                    'start': arg.range.start,
                    'end': arg.range.end
                })

            results['role'].append({
                'trigger': words[role.index],
                'index': role.index,
                'relation': relations
            })

        return results

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()
        self.recognizer.release()
        self.labeller.release()


def print_result(res):
    print('-------- basic info --------')
    for item in res['basic']:
        print(item['index'], '\t', item['word'], '\t', item['pos'], '\t', item['entity'], '\t', item['head'], '\t',
              item['relation'])

    print('-------- role info --------')
    for item in res['role']:
        print('trigger: ', item['trigger'])
        for rel in item['relation']:
            print('%s(%s, %s)' % (rel['name'], rel['start'], rel['end']))


if __name__ == '__main__':
    sentence = '辽宁本溪一铁矿发生炸药爆炸事故 已致11死9伤'
    ltp = LtpFormatter()
    res = ltp.format(sentence)
    print_result(res)

    ltp.release()
