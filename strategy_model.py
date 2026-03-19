# strategy_model.py
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pathlib import Path
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna
from qlib.utils import init_instance_by_config
from qlib.workflow import R     # [新增] 引入 Qlib 的核心工作流引擎，用于回测和记录
import pandas as pd
import numpy as np
from ruamel.yaml import YAML
import config
import logging
import warnings
import os      
import torch   

from qlib.data.dataset import TSDatasetH
warnings.filterwarnings("ignore")

class DeepGridStrategy:
    def __init__(self):
        print("-> [Strategy] 正在初始化 Qlib 引擎...")
        qlib.init(provider_uri=config.QLIB_DIR, region="cn", logging_level=logging.INFO)
        
        self.grid_step = 0.015
        self.trend_threshold = 0.02
        self.model = None
        self.weight_path = config.RESULTS_DIR+config.WEIGHT_PATH # 设定权重保存路径
        self.csv_filename = config.RESULTS_DIR

    def build_dataset_and_model(self, stock_list, start_date, end_date, is_training_day=False, yaml_path=config.YAML_PATH):
        """
        加载 YAML 配置，动态修改时间和股票池，然后根据日期决定是否训练 (用于实盘或单日步进)
        """
        print(f"-> [Strategy] 正在加载配置文件: {yaml_path}")
        
        yaml = YAML(typ="safe", pure=True)
        config_file = Path(yaml_path).absolute().resolve()
        task_config = yaml.load(config_file.open(encoding='utf-8'))
        
        dataset_config = task_config["task"]["dataset"]
        model_config = task_config["task"]["model"]
        
        try:
            test_segment = dataset_config["kwargs"]["segments"]["test"]
            start_str = pd.to_datetime(str(test_segment[0])).strftime('%Y%m%d')
            end_str = pd.to_datetime(str(test_segment[1])).strftime('%Y%m%d')
            self.csv_filename += f"{config.MODEL_NAME}_predictions_{start_str}_{end_str}.csv"
        except Exception as e:
            print(f"-> [警告] 提取测试日期失败，使用默认文件名。原因: {e}")
            self.csv_filename += f"{config.MODEL_NAME}_predictions_default.csv"

        print("-> [Strategy] 正在将动态参数注入配置...")
        handler_kwargs = dataset_config["kwargs"]["handler"]["kwargs"]
        handler_kwargs["instruments"] = "all"
        
        print("-> [Strategy] 正在实例化 Dataset 和 Model...")
        dataset = init_instance_by_config(dataset_config)
        self.model = init_instance_by_config(model_config)

        if is_training_day:
            print("-> [Strategy] 【周末任务】今天是周六，正在利用最新数据重新训练模型...")
            self.model.fit(dataset, save_path=self.weight_path)
            print(f"-> [Strategy] 模型重新训练完成！最新权重已保存至: {self.weight_path}")
            
        else:
            print("-> [Strategy] 【工作日任务】今天是交易日，跳过训练流程...")
            if os.path.exists(self.weight_path):
                print(f"-> [Strategy] 发现历史权重文件，正在极速加载: {self.weight_path}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.model.load_state_dict(torch.load(self.weight_path, map_location=device))
                self.model.fitted=True
            else:
                print("-> [警告] 未找到历史权重文件！触发紧急回退机制：正在强制执行初始训练...")
                self.model.fit(dataset, save_path=self.weight_path)
                print(f"-> [Strategy] 紧急训练完成，权重已保存至: {self.weight_path}")
        
        return dataset

    def get_model_prediction(self, dataset):
        print(f"-> [Strategy] 正在调用 {config.MODEL_NAME} 模型进行多维推理...")
        predictions = self.model.predict(dataset)
        
        if isinstance(predictions, pd.Series):
            pred_df = predictions.to_frame(name='pred_center_return')
        else:
            pred_df = predictions.copy()
            if 'pred_mu' in pred_df.columns:
                pred_df = pred_df.rename(columns={'pred_mu': 'pred_center_return'})
            else:
                pred_df.columns = ['pred_center_return']
                
        export_name = getattr(self, 'csv_filename', 'default_predictions.csv')
        pred_df.to_csv(export_name)    
        print(f"-> [Strategy] 多维预测结果已导出至: {export_name}")
        
        return pred_df

    def generate_actions(self, predictions_df:pd.DataFrame, current_prices, current_positions):
        print("-> [Strategy] 正在根据网格规则生成交易信号...")
        actions = {}
        
        for stock in predictions_df.index.get_level_values('instrument').unique():
            pred_return = predictions_df.loc[(slice(None), stock), 'pred_center_return'].iloc[-1]
            pred_return = float(pred_return)
            curr_price = current_prices.get(stock, 0)
            holding_vol = current_positions.get(stock, 0)
            
            if curr_price == 0:
                continue
                
            predicted_center = curr_price * (1 + pred_return)
            
            if pred_return > self.trend_threshold:
                actions[stock] = {"action": "BUY", "target_price": curr_price, "reason": "Long Trend"}
            elif pred_return < -self.trend_threshold:
                if holding_vol > 0:
                    actions[stock] = {"action": "SELL", "target_price": curr_price, "reason": "Short Trend Panic"}
            else:
                buy_grid_line = predicted_center * (1 - self.grid_step)
                sell_grid_line = predicted_center * (1 + self.grid_step)
                
                if curr_price <= buy_grid_line:
                    actions[stock] = {"action": "BUY", "target_price": buy_grid_line, "reason": "Grid Buy"}
                elif curr_price >= sell_grid_line and holding_vol > 0:
                    actions[stock] = {"action": "SELL", "target_price": sell_grid_line, "reason": "Grid Sell"}
                else:
                    actions[stock] = {"action": "HOLD", "target_price": None, "reason": "In Grid Center"}
                    
        return actions

    # =====================================================================
    # 【新增功能】：完全脱离命令行，利用 YAML 在本地直接启动完整的论文复现回测
    # =====================================================================
    def run_full_backtest(self, yaml_path=config.YAML_PATH):
        print(f"\n==================================================")
        print(f"🌟 启动完整 YAML 回测工作流: {yaml_path}")
        print(f"==================================================")
        yaml = YAML(typ="safe", pure=True)
        task_config = yaml.load(Path(yaml_path).absolute().open(encoding='utf-8'))["task"]

        print("-> [Workflow] 1. 初始化 Dataset 和 Model...")
        dataset = init_instance_by_config(task_config["dataset"])
        model = init_instance_by_config(task_config["model"])

        # 启动 Qlib 原生实验记录器 (去掉 as rec)
        with R.start(experiment_name=f"{config.MODEL_NAME}_Backtest"):
            
            # 【核心修复】：在这里调用 get_recorder() 获取真正的记录器对象
            rec = R.get_recorder()
            
            print("\n-> [Workflow] 2. 开始在全量历史数据上训练模型 (Model Fit)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.model.load_state_dict(torch.load(self.weight_path, map_location=device))
            model.fitted=True
            #model.fit(dataset)
            
            # 全局 R 可以直接用来存 model
            R.save_objects(trained_model=model)

            print("\n-> [Workflow] 3. 开始执行记录器与模拟交易流程 (Records)...")
            for record_conf in task_config["record"]:
                
                # 【核心修复】：把真正的记录器 rec 传给组件
                record_conf["kwargs"]["recorder"] = rec
                
                # 动态将刚实例化的 model 和 dataset 注入到配置中
                if record_conf["kwargs"].get("model") == "<MODEL>":
                    record_conf["kwargs"]["model"] = model
                if record_conf["kwargs"].get("dataset") == "<DATASET>":
                    record_conf["kwargs"]["dataset"] = dataset

                record_class_name = record_conf['class']
                print(f"\n>>> 正在运行分析组件: [{record_class_name}] <<<")
                
                try:
                    record_task = init_instance_by_config(record_conf)
                    record_task.generate()
                    print(f"[*] {record_class_name} 执行成功！")
                except Exception as e:
                    print(f"[-] {record_class_name} 执行时发生错误: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print("\n==================================================")
        print("🏆 论文复现全流程回测执行完毕！")
        print("回测详细数据、资金曲线、以及模型文件均已保存至本目录下的 `mlruns/` 文件夹中。")
        print("==================================================")


def run_strategy(stock_list, today_str, current_prices, current_positions):
    strategy = DeepGridStrategy()
    current_date = pd.to_datetime(today_str)
    day_of_week = current_date.dayofweek
    is_training_day = (day_of_week == 5)
    start_date = (current_date - pd.Timedelta(days=60)).strftime('%Y%m%d')
    
    dataset = strategy.build_dataset_and_model(
        stock_list=stock_list, 
        start_date=start_date, 
        end_date=today_str, 
        is_training_day=is_training_day
    )
    
    predictions = strategy.get_model_prediction(dataset)
    actions = strategy.generate_actions(predictions, current_prices, current_positions)
    return actions


if __name__ == '__main__':
    import sys
    import json
    import traceback
    
    # ---------------------------------------------------------
    # 启动模式分流：如果你在终端输入 `python strategy_model.py backtest`
    # 就会执行完整的 YAML 回测。否则执行单日调试。
    # ---------------------------------------------------------
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'backtest':
        strategy = DeepGridStrategy()
        strategy.run_full_backtest(config.YAML_PATH)
        sys.exit(0)

    print("=== 开始本地独立调试 strategy_model ===")
    
    test_stock_list = ['SZ000001', 'SH600050']
    test_today_str = '20231027' 
    test_current_prices = {'SZ000001': 10.50, 'SH600050': 8.20}
    test_current_positions = {'SZ000001': 1000, 'SH600050': 0}

    print(f"[*] 调试日期: {test_today_str} (星期 {pd.to_datetime(test_today_str).dayofweek + 1})")
    print(f"[*] 调试标的: {test_stock_list}")

    try:
        actions = run_strategy(
            stock_list=test_stock_list,
            today_str=test_today_str,
            current_prices=test_current_prices,
            current_positions=test_current_positions
        )
        print("\n=== 调试成功！输出的交易动作 (Actions) ===")
        print(json.dumps(actions, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"\n[-] [调试报错] 策略运行失败: {str(e)}")
        traceback.print_exc()
        
    print("\n=== 本地调试结束 ===")
    print("提示：如果你想运行 YAML 完整回测，请在终端执行: python strategy_model.py backtest")