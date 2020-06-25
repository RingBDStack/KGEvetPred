"""
@author: Li Xi
@file: ltp_analyzer.py
@time: 2020/1/14 20:20
@desc:
"""

from pyltp import SentenceSplitter

from utils.ltp_formatter import LtpFormatter
# 本文件中的方法是根据LtpFormatter的结果，进行进一步的分析应用
from utils.tools import modify_sentence

ltp = LtpFormatter()  # 初始化LtpFormatter


def single_ltp(sentence, analyse_need='total'):
    """
    获取单个ltp分析结果
    :param sentence: 句子字符串
    :param analyse_need: 标柱需要的分析范围，可选值有两个
        total：获得全部的ltp分析结果
        pos：获得ltp的分词结果和词性标注结果
    :return: ltp的分析结果，结果格式参考tools.ltp_formatter中的格式
    """
    if analyse_need == 'total':
        return ltp.format(sentence)
    elif analyse_need == 'pos':
        return ltp.format_only_pos(sentence)


def get_key_sentences(content):
    """
    寻找新闻正文中的重点句
    首先对新闻正文进行分句，然后根据ltp分析结果，找到含有时间词的句子
    :param content: 待分析的新闻内容
    :return: 重点句list
    """
    if isinstance(content, list):
        ss = content
    else:
        # 分句
        if content.strip() == '':
            return ''
        ss = SentenceSplitter.split(content)

    sentences = []
    for s in ss:
        s = s.strip()
        if s != '':
            sentences.append(str(s))

    # 这里只需要获取ltp分析中的pos结果即可（包含时间标注）
    # analyses = []
    # for sentence in sentences:
    #     analyses.append(single_ltp(sentence, analyse_need='pos'))

    # 对结果进行整理
    key_sentences = []
    for j in range(len(sentences)):
        sentence = sentences[j]
        # tmp_analyse = analyses[j]
        # cnt = 0
        # for w in tmp_analyse['basic']:
        #     if w['pos'] == 'nt':
        #         cnt += 1
        # if cnt > 1:
        # tmp_sentence = modify_sentence(sentence)  # 去除句子中的噪声
        # if len(tmp_sentence) > 0:

        # 添加句子去重
        sentence =  modify_sentence(sentence)
        if sentence not in key_sentences:
            key_sentences.append(sentence)

    return key_sentences


def get_entity_patterns(analy, entity_type, start_pos, end_pos):
    """
    获取句子中的命名实体
    :param analy: 待分析句子的ltp分析结果
    :param entity_type: 需要获取的命名实体类别，有三种类别分别是
        'Nh' -- person, 'Ni' -- Organization, 'Ns' -- Location
    :param start_pos: 分析的开始位置
    :param end_pos: 分析的结束位置
    :return: 返回命名实体list

    其他标签说明：
    O：非命名实体
    S：单个词是命名实体
    B：词是命名实体的开始
    I：词是一个命名实体的中间的词
    E：词在命名实体的最后

    示例：S-Ni 这个词是一个机构名

    """
    ret_entity = set()
    entity_pattern = ''

    for i in range(start_pos, end_pos + 1):
        w = analy['basic'][i]
        if (w['entity'] == 'B-' + entity_type) or (w['entity'] == 'B-' + entity_type):
            entity_pattern += w['word']
        elif (w['entity'] == 'E-' + entity_type) or (w['entity'] == 'S-' + entity_type):
            entity_pattern += w['word']
            ret_entity.add(entity_pattern)
            entity_pattern = ''

    return list(ret_entity)


def trigger_rank(analyse, trigger_index):
    '''
    计算trigger rank，句法依赖关系中作为head出现的次数，即作为节点的出度
    :param analyse: ltp分析结果
    :param trigger_index: 待计算的trigger 的 index
    :return: rank 结果
    '''
    cnt = 0
    for word in analyse['basic']:
        if word['head'] == trigger_index:
            cnt += 1
    return cnt


def get_relate_pattern(analyse, parent_index, pattern_type, ret_pattern):
    '''
    递归的方法获取完整的关系树，主要针对"ATT"关系
    :param analyse: ltp分析结果
    :param parent_index: parent的index
    :param pattern_type: 需要获得的关系种类，ATT
    :param ret_pattern: 存储pattern结果
    :return:
    '''
    for word in analyse['basic']:
        if word['head'] == parent_index and word['relation'] == pattern_type:
            ret_pattern.add((word['word'], word['index']))
            ret_pattern |= get_relate_pattern(analyse, word['index'], pattern_type, ret_pattern)
    return ret_pattern


def get_trigger_pattern(analyse, trigger_index):
    '''
    找到trigger及其补充信息
    到第一个ATT/None停止
    Trigger -> SBV -> ATT   （1）
    Trigger -> VOB -> ATT   （2）
                |---> SBV -> ATT   （3）
                |---> COO -> ATT   （4）
                ----> COO -> SBV -> ATT   （5）
    :param analyse: 句子ltp分析结果
    :param trigger_index: trigger 的index
    :return: trigger pattern trigger及其补充信息
    '''

    # VOB相关信息
    ret_pattern = set()
    ret_pattern.add((analyse['basic'][trigger_index]['word'], trigger_index))

    for word in analyse['basic']:
        if word['head'] == trigger_index and word['relation'] == 'VOB':  # 谓宾关系
            ret_pattern.add((word['word'], word['index']))
            ret_pattern |= get_relate_pattern(analyse, word['index'], 'ATT', ret_pattern)

            for word1 in analyse['basic']:
                if word1['head'] == word['index'] and word1['relation'] == 'SBV':
                    ret_pattern.add((word1['word'], word1['index']))
                    ret_pattern |= get_relate_pattern(analyse, word1['index'], 'ATT', ret_pattern)

            for w in analyse['basic']:
                if w['head'] == word['index'] and w['relation'] == 'COO':  # 并列关系
                    ret_pattern.add((w['word'], w['index']))
                    ret_pattern |= get_relate_pattern(analyse, w['index'], 'ATT', ret_pattern)
                    for word1 in analyse['basic']:
                        if word1['head'] == w['index'] and word1['relation'] == 'SBV':
                            ret_pattern.add((word1['word'], word1['index']))
                            ret_pattern |= get_relate_pattern(analyse, word1['index'], 'ATT', ret_pattern)

    if len(ret_pattern) > 1:
        ret_pattern = sorted(list(ret_pattern), key=lambda x: x[1])
        first_index = ret_pattern[0][1]
        last_index = ret_pattern[-1][1]
        ret_pattern = [x[0] for x in ret_pattern]
        ret_pattern = ''.join(ret_pattern)
        return ret_pattern, first_index, last_index

    # SBV相关
    ret_pattern = set()
    ret_pattern.add((analyse['basic'][trigger_index]['word'], trigger_index))

    for word in analyse['basic']:
        if word['head'] == trigger_index and word['relation'] == 'SBV':  # 主谓关系
            ret_pattern.add((word['word'], word['index']))
            ret_pattern |= get_relate_pattern(analyse, word['index'], 'ATT', ret_pattern)

    ret_pattern = sorted(list(ret_pattern), key=lambda x: x[1])
    first_index = ret_pattern[0][1]
    last_index = ret_pattern[-1][1]
    ret_pattern = [x[0] for x in ret_pattern]
    ret_pattern = ''.join(ret_pattern)
    return ret_pattern, first_index, last_index


if __name__ == '__main__':
    news_content = "北京时间2017年6月22日凌晨5点左右，在浙江杭州蓝色钱江小区2幢1单元1802室发生纵火案。" \
                   "该事件造成4人死亡（一位母亲和三个未成年孩子）。" \
                   "2017年7月1日，根据杭州市人民检察院批准逮捕决定，杭州市公安局对涉嫌放火罪、盗窃罪的犯罪嫌疑人莫焕晶依法执行逮捕。" \
                   "2017年8月21日，杭州市检察院依法对被告人莫焕晶提起公诉。" \
                   "2017年12月21日上午9时许，杭州“蓝色钱江放火案”在浙江省杭州市中级人民法院公开开庭审理。法庭宣布延期审理。" \
                   "2017年12月25日，杭州市公安消防局再次收到受害人家属林某某提出的政府信息公开申请，杭州市公安消防局局将根据《中华人民共和国政府信息公开条例》的有关规定，在法定期限内做出答复。" \
                   "2018年2月9日，案件一审公开宣判，被告人莫焕晶被判死刑。 2月24日，从浙江省高级人民法院获悉，莫焕晶已向该院提起上诉。" \
                   "5月17日9时，莫焕晶上诉开庭审理；下午17时20分许，庭审结束，审判长宣布择期宣判。 " \
                   "6月4日，案件作出二审裁定：驳回上诉，维持原判。" \
                   "2018年9月21日，经最高人民法院核准，莫焕晶被执行死刑。"

    key_sentence = get_key_sentences(news_content)

    for ks in key_sentence:
        print(ks)
