# KGEvetPred


This project mainly includes two major tasks:
* Event Evolution Knowledge Graph Generation
* Event prediction



## Event Evolution Knowledge Graph Generation

### Description

This task first analyzes the news data, extracts the event chain and event elements from it, and finally generates an event graph.


![image](pictures/example.png)

### Files

* `/corpus`：stores raw data
* `/front`：stores front-end related code
    * more details in [front-end readme.md](front/readme.md)
* `/utils`：stores generation event tools. There are detailed comments in this part of the file, you can directly open the file to view the function description.
    * ltp analyzer 
    * ltp formatter 
    * graph manager 
    * tools 

### Precautions

There are still some imperfections in the event analysis part, and TODO is used to mark the part that can be improved.

## Event prediction

### Description

1. Generate experimental data:
     * In `/corpus` is the generated experimental data
     * `/scripts/make_dataset.ipynb` is the script to generate experimental data
2. Model
     * `/models` contains model documents
     * There are 7 models in total
     * Model training can use `/train.py`
     * `/start.sh` is to use the server to run the model training script
3. Visualization of results
     * `/scripts/draw.py` is a script for visualizing model training results, `/scripts/getscore.py` is a script for obtaining model score data, which is used to assist drawing

### Precautions

1. Create a log folder in the running directory
2. When testing the model (in the model code file) in a single file, you need to comment out the register line

## Others

1. Many files are not uploaded on github, you can find me to copy them directly, please refer to `/.gitignore`
2. The required packages are in `/requirements.txt`
3. `/my_logger.py` is the logger script, and the log is stored in `/log`

# 基于事理图谱的事件预测（KGEvetPred）


本项目主要包含两个大任务，分别是：
* 事理图谱抽取
* 事件预测



## 事理图谱抽取

### 主要任务描述

分析新闻数据，抽取事件链和事件要素，最终生成事理图谱。


![image](pictures/example.png)

### 文件介绍

该部分文件主要包括：
* `/corpus`：主要存储了原始数据
* `/front`：主要存储了前端相关代码
    * 前端相关可在[readme.md](front/readme.md)中查看
* `/utils`：主要存储了生成事件工具
    * 该部分文件中都有详细的注释，可直接打开文件查看功能描述
    * ltp analyzer 是ltp分析工具
    * ltp formatter 是将ltp分析的结果格式化
    * graph manager 是生成时事件图谱的工具
    * tools 一些通用的功能

### 注意事项

事件分析部分还有一些不完善的地方，用TODO标注了能够提升的部分。

## 事件预测

### 主要任务/文件描述

1. 生成实验数据：
    * `/corpus`中是生成好的实验数据
    * `/scripts/make_dataset.ipynb`中是生成实验数据的脚本
2. 模型
    * `/models`里面是模型文档
    * 一共有7中模型
    * 模型训练可以使用`/train.py`
    * `/start.sh`是使用服务器运行模型训练脚本
3. 结果可视化
    * `/scripts/draw.py`里面是模型训练结果可视化的脚本，`/scripts/getscore.py`是获取模型score数据的脚本，用来辅助绘图

### 注意事项

1. 在运行目录创建log文件夹
2. 单文件测试model（在模型代码文件中） 时需要将register那一行注释掉

## 其他

1. 很多文件没有上传在github上，可以直接找我拷贝，可参考`/.gitignore`
2. `/requirements.txt`中是需要的包
3. `/my_logger.py`是logger脚本，日志存储在`/log`中
