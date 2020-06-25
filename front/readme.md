
# README

## Usage

* 使用`util/graph_manager`生成`graph.json`
* 使用`front/makeXML.py`生成需要xml格式文件
* 使用`index.html`查看可视化结果(浏览器会限制访问本地文件，推荐使用Firefox查看html文件)
* 需要使用项目中自带的dataTool.js(其他的最好也用项目中自带的)


## Problems and Solutions

1. 本地Ajax跨域报错Cross origin requests are only supported for protocol schemes: http, data, chrome, chrome-extension, https. 

```
使用firefox浏览器查看前端界面
```

2. 如何配置Firefox火狐浏览器访问本地文件

* 进入火狐的配置界面: 在浏览器搜索框输入`about:config`
* 点击”我了解此风险”后进入页面 
* 搜索”security.fileuri.strict_origin_policy”, 并设置该项为false 
