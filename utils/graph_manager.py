"""
@author: Li Xi
@file: graph_manager.py
@time: 2020/1/14 20:22
@desc:

构造事件图谱：
1. 分析事件：分析事件触发词和arguments（时间、地点、施事者、受事者、机构等）
2. GraphController 构造、存储事件图谱

"""


import json
from utils.ltp_analyzer import get_key_sentences, get_entity_patterns, single_ltp, trigger_rank, get_trigger_pattern
from utils.ltp_formatter import print_result
from utils.tools import check_trigger_list


class GraphController(object):
    '''
    构造事件图谱
    '''

    def __init__(self):
        self.graph = {}

    def add_edge(self, start, end, info):
        '''
        添加一条边
        :param start: start trigger
        :param end: end trigger
        :return:
        '''

        if start is None or end is None or info is None:
            return

        if start not in self.graph.keys():
            self.graph[start] = {}
            self.graph[start][end] = [info]
        else:
            if end in self.graph[start].keys():
                # print('ERROR: 出现冲突！！！！')
                # print('------', start, end, info, self.graph[start][end])
                self.graph[start][end].append(info)
                return
            else:
                self.graph[start][end] = [info]

    def add_child_list(self, parent, child_list, info):
        '''
        添加一个parent和所有child之间的边
        :param parent: parent node
        :param child_list: child node list
        :param info: 边的信息/类型
        '''
        for item in child_list:
            self.add_edge(parent, item, info)

    def add_trigger_list(self, trigger_list, info):
        '''
        添加trigger list之间相邻trigger之间的边的信息
        :param trigger_list: trigger list
        :param info: 边的信息/类型
        '''
        for i in range(1, len(trigger_list)):
            self.add_edge(trigger_list[i - 1], trigger_list[i], info)


def get_news_graph(graph_controller, news_content):
    '''
    构造news content的graph信息
    :param graph_controller: GraphController实例，存储图的信息
    :param news_content: news 内容
    :return: GraphController实例，存储图的信息
    '''

    key_sentence = get_key_sentences(news_content)  # 分句

    root_trigger_list = []

    for sentence in key_sentence:

        analyse = single_ltp(sentence)  # ltp
        # print_result(analyse)

        # TODO: 获取句子时间信息方法改善，当简单的使用第一个逗号前面的部分作为时间信息
        event_time = None
        # if '，' in sentence:
        #     event_time = sentence.split('，')[0]

        # 获取句子位置信息
        event_position = get_entity_patterns(analy=analyse,
                                             entity_type='Ns',
                                             start_pos=0,
                                             end_pos=len(analyse['basic']) - 1)

        # 获取句子机构信息全句
        event_organization = get_entity_patterns(analy=analyse,
                                                 entity_type='Ni',
                                                 start_pos=0,
                                                 end_pos=len(analyse['basic']) - 1)

        sentence_trigger_list = []
        for role in analyse['role']:
            # 判定为不重要的trigger 不予考虑
            # if trigger_rank(analyse, role['index']) <= 1:
            #     continue

            # 获取句子机构信息（A0，A1）
            # 获取句子人物信息（A0，A1）
            event_organization_in_a0 = []
            event_organization_in_a1 = []
            event_person_in_a0 = []
            event_person_in_a1 = []
            graph_controller.add_child_list(trigger_pattern, [event_time], 'time')
            for relation in role['relation']:
                if relation['name'] == 'A0':
                    event_organization_in_a0 += get_entity_patterns(analy=analyse,
                                                                    entity_type='Ni',
                                                                    start_pos=relation['start'],
                                                                    end_pos=relation['end'])
                    event_person_in_a0 += get_entity_patterns(analy=analyse,
                                                              entity_type='Nh',
                                                              start_pos=relation['start'],
                                                              end_pos=relation['end'])
                if relation['name'] == 'A1':
                    event_organization_in_a1 += get_entity_patterns(analy=analyse,
                                                                    entity_type='Ni',
                                                                    start_pos=relation['start'],
                                                                    end_pos=relation['end'])
                    event_person_in_a1 += get_entity_patterns(analy=analyse,
                                                              entity_type='Nh',
                                                              start_pos=relation['start'],
                                                              end_pos=relation['end'])

            # 分析trigger元素，基于trigger role以及其他的句法分析结果：ATT、VOB、COO、SVB等关系
            trigger_pattern, first_index, last_index = get_trigger_pattern(analyse, role['index'])
            sentence_trigger_list.append((trigger_pattern, first_index, last_index))

            # 将trigger相关的人名、机构名信息加入到graph中
            graph_controller.add_child_list(trigger_pattern, event_organization_in_a0, 'organization-A0')
            graph_controller.add_child_list(trigger_pattern, event_organization_in_a1, 'organization-A1')
            graph_controller.add_child_list(trigger_pattern, event_person_in_a0, 'person-A0')
            graph_controller.add_child_list(trigger_pattern, event_person_in_a1, 'person-A1')

            # 判断是不是root trigger
            if analyse['basic'][role['index']]['head'] == -1:
                # 如果是root trigger，加入到root trigger list中，并将时间、地点、机构名信息加入到graph中
                root_trigger_list.append(trigger_pattern)
                graph_controller.add_child_list(trigger_pattern, event_position, 'position')
                graph_controller.add_child_list(trigger_pattern, event_organization, 'organization')

        # sentence_trigger_list 进行排序
        sentence_trigger_list = check_trigger_list(sentence_trigger_list)
        sentence_trigger_list = [x[0] for x in sentence_trigger_list]

        # 将sentence trigger list加入到graph中
        # TODO：更改边添加的方式
        graph_controller.add_trigger_list(sentence_trigger_list, 'next')

    # 将root trigger list加入到graph中
    graph_controller.add_trigger_list(root_trigger_list, 'next')

    return graph_controller


if __name__ == '__main__':
    news_content = "北京时间2017年6月22日凌晨5点左右，在浙江杭州蓝色钱江小区2幢1单元1802室发生纵火案。" \
                   "该事件造成4人死亡（一位母亲和三个未成年孩子）。" \
                   "2017年7月1日，根据杭州市人民检察院批准逮捕决定，杭州市公安局对涉嫌放火罪、盗窃罪的犯罪嫌疑人莫焕晶依法执行逮捕。" \
                   "2017年8月21日，杭州市检察院依法对被告人莫焕晶提起公诉。" \
                   "2017年12月21日上午9时许，杭州“蓝色钱江放火案”在浙江省杭州市中级人民法院公开开庭审理。法庭宣布延期审理。" \
                   "2017年12月25日，杭州市公安消防局再次收到受害人家属林某某提出的政府信息公开申请，杭州市公安消防局局将根据《中华人民共和国政府信息公开条例》的有关规定，在法定期限内做出答复。" \
                   "2018年2月9日，案件一审公开宣判，被告人莫焕晶被判死刑。 " \
                   "2月24日，从浙江省高级人民法院获悉，莫焕晶已向该院提起上诉。" \
                   "5月17日9时，莫焕晶上诉开庭审理；下午17时20分许，庭审结束，审判长宣布择期宣判。 " \
                   "6月4日，案件作出二审裁定：驳回上诉，维持原判。" \
                   "2018年9月21日，经最高人民法院核准，莫焕晶被执行死刑。"
    graph_counter = GraphController()
    news_graph_count = get_news_graph(graph_counter, news_content)
    with open("../front/data/graph.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(news_graph_count.graph, ensure_ascii=False))

