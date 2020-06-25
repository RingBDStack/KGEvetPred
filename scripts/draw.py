"""
@author: Li Xi
@file: draw.py
@time: 2020/6/11 13:49
@desc:
"""

import json
import os

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc



# 修改model的绘制顺序
models = ["DA-EvtLSTM", "EvtLSTM", "MemNet", "GMN", "ATAE", "GRMN"]
# 修改对应model的输出的名字，和上面的顺序对应
model_output_name = ["DA-EvtLSTM", "EvtLSTM", "MemNet", "GMN", "ATAE", "GRMN"]
# 修改颜色，对应model的顺序
colors = ['#4169E1', 'b', '#1E90FF', 'orange', '#FAA460', '#FF4500', 'turquoise']


# 事件类型可选
event_types = [-1, 0, 1, 2, 3, 4]
# batch size可选
batch_sizes = [32, 64]

# 选择额的batch size和event type，和上面的对应
bs = batch_sizes[0]
et = event_types[2]

output_path = "output"
output_files = os.listdir(output_path)


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,

         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13, }


lw = 2
plt.figure(figsize=(9, 6), facecolor=(1, 1, 1))

for i in range(len(models)):
    model_name = models[i]

    cur_output = sorted([x
                  for x in output_files if
                  model_name == x[:len(model_name)] and
                  "bs_" + str(bs) in x and
                  "et_" + str(et) in x
                  ])
    print("\n".join(cur_output))
    print('---------------------')
    # continue
    # 这里可选的有5个输出结果，默认选取第一个 cur_output[0]
    # 按照时间顺序排列，有一些效果不好的可以调整这里，选择其他的数据
    with open(os.path.join(output_path, cur_output[0]), 'r', encoding='utf-8') as f:
        content = json.loads(f.read())


    label_y = content['label_y']
    scores = content['scores']
    precisions, recalls, _tresholds = precision_recall_curve(label_y, scores, pos_label=1)
    auc_ret = auc(recalls, precisions)

    plt.plot(recalls, precisions, color=colors[i], lw=lw,
         label='%s (AUC = %6f)' % (model_output_name[i], auc_ret))  # TODO: change model name

# exit(0)
# ----------- plot settings -----------
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(fontsize=15, **font2)
plt.yticks(fontsize=15, **font2)
plt.xlabel('Recall', fontsize=20, **font1)
plt.ylabel('Precision', fontsize=20, **font1)
# plt.title('Plot Title')
plt.legend(loc="lower left", prop=font2)
plt.grid(c='grey', linestyle='--')
plt.show()

# save as png
fig = plt.gcf()
# 保存文件
fig.savefig('pr_curve.png', dpi=100)
