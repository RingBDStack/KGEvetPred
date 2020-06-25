#!/usr/bin/env bash
#SBATCH -c 6
#SBATCH -p dell
#SBATCH --gres=gpu:1
#
# -m: 模型选择 0 ~ 7
# -fd: 数据正负比 1, 3, 4
# -th: threshold 0.3 0.4 0.5 0.6 0.7
# -et: event type 事件种类 -1（不分类）, 0, 1, 2, 3, 4
# 此处k表示实验重复次数

rm tensorboard/*.th tensorboard/*.json

for k in 0 1 2 3 4
do
    for i in -1 0 1 2 3 4
    do
        for j in  0 1 2 3 4 6
        do
            python run.py -m $j -emb 300 -lr 1e-3 -bs 64 -pc 5 -ep 50 -hs 32 -hn 3 -fd 4 -th 0.5 -et $i
            sleep 1

            # remove tmp model info
            #cd tensorboard/log
            #mkdir $j-fd-4-et-$i
            #cd ../..
            #mv tensorboard/*.th tensorboard/log/$j-fd-4-et-$i
            #mv tensorboard/*.json tensorboard/log/$j-fd-4-et-$i
            #sleep 1
        done
    done
done

