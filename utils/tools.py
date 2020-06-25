"""
@author: Li Xi
@file: tools.py
@time: 2020/1/13 23:26
@desc:
"""

import re

# cut word 及之前容易出现非核心语义信息，如：新华社发布等
cut_words = ['指的是', '日讯', '称，', '日说', '报道说', '报道', '获悉', '日电', '消息', '了解到', '快讯', ' ', '\t']


def modify_sentence(sentence):
    """
    去除句子噪声，保留句子的核心语义信息。
    噪声包括：与事件内容无关的机构信息、地点信息等，例如："新华社发布..."等信息
    :param sentence: 待处理的句子
    :return: 除去噪声信息的句子
    """
    sentence = re.sub("\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\(.*?\)", "", sentence)

    cut_clear = False
    while cut_clear is False:
        cut_clear = True
        for cut_word in cut_words:
            if sentence.find(cut_word) != -1:
                cut_clear = False
                sentence = sentence[sentence.find(cut_word) + len(cut_word):].strip()
                if len(sentence) > 0:
                    if sentence[0] == '，' or sentence[0] == ',':
                        sentence = sentence[1:]
                break

    if len(sentence) < 5:  # 过滤到过短的句子
        return ''

    return sentence


def check_trigger_list(trigger_list):
    '''
    对trigger list进行排序，（去重）
    :param trigger_list: 待处理的trigger list
    :return: 处理好的trigger list
    '''
    ret_trigger_list = trigger_list

    # 对trigger list 去重，去除中间包含关系的trigger关系
    for trigger in trigger_list:
        for i in range(len(ret_trigger_list)):
            # trigger start 小 trigger end 大
            if ret_trigger_list[i][1] >= trigger[1] and ret_trigger_list[i][2] <= trigger[2]:
                ret_trigger_list[i] = trigger

    ret_trigger_list = sorted(ret_trigger_list, key=lambda x: x[2])

    return ret_trigger_list


