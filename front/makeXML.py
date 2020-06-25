"""
@author: Li Xi
@file: makeXML.py
@time: 2020/2/6 15:57
@desc:
"""

import json
import xml.dom.minidom


def json2graph(file_path):
    """
    从graph.json中获取node信息和edge信息:
        node_list = [
            {
                "name":,
                "type":,
                "id":,
            }, ......
        ]
        edge_list = [
            {
                "source":,
                "target":,
            }, ......
        ]
        !!!!!!说明： 为了node分为两种 trigger和attribute，对二者分别编号
        node是唯一的，编号为e1，e2，e3....
        为了使前端显示更加清晰，对每个trigger的attibute进行区分，即使内容一样，编号为a1, a2, a3......
    :param file_path:
    :return: node_list, edge_list
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.loads(f.read())

    node_list = []
    trigger_dict = dict()
    attribute_cnt = 0
    edge_list = []

    for source in content.keys():
        # source node trigger 需要保证唯一性
        if source not in trigger_dict.keys():

            trigger_dict[source] = "e{}".format(str(len(trigger_dict)))

            node_list.append({
                "name": source,
                "type": "trigger",
                "id": trigger_dict[source]
            })

        # 用于edge的添加
        source_id =trigger_dict[source]

        # 遍历source的下一节点
        for target in content[source]:
            # TODO：目前是取第一个作为value
            value = content[source][target][0]
            # 将后面的A0/A1去除
            node_type = value.split('-')[0]

            # 如果是next trigger
            if node_type == "next":
                node_type = "trigger"
                node_name = target
                # 需要保证唯一性
                if node_name not in trigger_dict.keys():
                    trigger_dict[node_name] = "e{}".format(str(len(trigger_dict)))
                    node_list.append({
                        "name": node_name,
                        "type": node_type,
                        "id": trigger_dict[node_name],
                    })
                target_id = trigger_dict[node_name]
            else:
                # 如果是attribute不需要保证唯一性（time，person，organization，position等）
                node_name = "{}: {}".format(value, target)
                node_type = "attribute"
                target_id = "a{}".format(str(attribute_cnt))
                attribute_cnt += 1
                node_list.append({
                    "name": node_name,
                    "type": node_type,
                    "id": target_id
                })

            # 添加edge
            edge_list.append({
                "source": source_id,
                "target": target_id
            })
    return node_list, edge_list


def write_xml(node_list, edge_list):
    """

    :param node_list:
    :param edge_list:
    :return:
    """
    doc = xml.dom.minidom.Document()

    root = doc.createElement("gexf")
    doc.appendChild(root)

    graph = doc.createElement("graph")
    root.appendChild(graph)

    nodes = doc.createElement("nodes")
    edges = doc.createElement("links")
    graph.appendChild(nodes)
    graph.appendChild(edges)

    for item in node_list:
        node = doc.createElement("node")
        node.setAttribute("name", item["name"])
        node.setAttribute("type", item["type"])
        node.setAttribute("id", item["id"])
        nodes.appendChild(node)

    for item in edge_list:
        edge = doc.createElement("edge")
        edge.setAttribute("source", item["source"])
        edge.setAttribute("target", item["target"])
        edges.appendChild(edge)

    with open("data/graph.xml", "w", encoding="utf-8") as f:
        doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')


if __name__ == '__main__':
    file_path = 'data/graph.json'
    # file_path = '../news/output/机票价格涨价为哪般？.json'
    node_list, edge_list = json2graph(file_path)
    print("Get {} nodes and {} edges...".format(str(len(node_list)), str(len(edge_list))))
    write_xml(node_list, edge_list)
    print('Done!!!')
