# 环境配置

``````bash
conda create -n quant python=3.10
conda activate quant
pip install xtquant
pip install pyqlib
pip install torch==2.8.0
pip install fastapi uvicorn
``````

# 构成说明

由于xtquant不能在linux服务器下跑，但模型训练需要服务器下的gpu，所以该项目下的部分文件仅用于其中某个系统

### qlib-main/

qlib项目源代码库，用于调用其中的数据处理文件dump_bin.py以及其内置模型

### :star:FactorVAE/

来自https://github.com/x7jeon8gi/FactorVAE项目，修改后添加了batch输入进行训练，最后回测结果良好，目前计划基于此进行策略的构建。

### :star: model_code/

基于qlib框架实现的代码及yaml配置放在该文件夹里

### config.py

配置文件，可以修改一下路径什么的

### csv2bin.bat

> windows

Windows下调用dump_bin.py的脚本文件

### plot_return.py

将预测出的csv的结果和真实市场数据可视化，画出折线图

### check_real_volatility.py

拉取真实市场数据后将其按照特定格式导出至csv文件中

### data_process.py

> windows

执行选股算法，并下载选出股票的数据到本地

### lstm_config.yaml

训练lstm模型所需的数据集、模型配置文件

### selector.py

> windows

根据多因子选择活跃的top_n股票

### server_api.py

> linux

在linux服务器端启用api服务，在实盘交易时接收windows端传来的当天数据，调用策略模型得到action，传回到windows端

### start_server.sh

> linux

启动server_api，开始监听请求

### :star: strategy_model.py

> linux

训练深度学习模型，使用模型进行预测，以及网格算法。

### trader.py

> windows

连接QMT国金系统、向linux服务器发送当天数据并得到给出的预测和action，根据action在QMT系统进行交易

### back_test.py

> 将代码复制到QMT策略编辑器中

回测代码，读取csv再用网格算法得到action

# 运行说明

配置好环境后，在QMT平台上下载好所需数据，可以参考https://zhuanlan.zhihu.com/p/1987887250613228523，windows上运行```python ./data_process.py```，得到**qlib_source_csvs/ **文件夹，内部是每个股票的csv文件；然后运行```./csv2bin.bat```处理得到**qlib_data/ **文件夹，包含calendars/、features/、instruments/文件夹。在服务器上运行```python ./strategy_model.py```，会通过一个简单的测试用的main函数第一次调用策略，训练出模型**lstm_model_weights.pth**，并自行进行预测，LSTM模型预测结果会保存到lstm_predictions_2019_2024.csv中（代码写的烂，main函数中测试的两个股票最后没有输出是正常的，能跑通得到这两个文件即可）。

其他涉及到api交互等代码暂时用不到，目前是回测阶段

# 回测说明

back_test.py中涉及到csv文件的路径，需要先把csv文件复制到windows上然后修改路径名指向该文件。

将back_test.py代码复制到QMT的策略编辑器中，然后右侧可以配置一些参数，再点回测应该就可以正常跑通。

![image](./asset/image.png) 

# TODO
总体思路是深度学习+网格算法，然后可以加入一些硬性的规则约束，比如总收益达到多少就可以全部卖出获得利润这种。目前回测代码中实现的是LSTM预测当天的涨跌率，然后基于昨天的收盘价得到当天的网格中枢，然后基于此建立网格进行买卖的决策。
总的回测框架差不多搭好了，就是起码可以跑通了，现在就是打算去优化模型，包括深度学习预测模型、网格算法、选股算法、规则约束这种。
