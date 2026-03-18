# strategy_model.py
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pathlib import Path
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna
from qlib.utils import init_instance_by_config
import pandas as pd
import numpy as np
from ruamel.yaml import YAML
import config
import logging
import warnings
import os      # [新增] 用于检查文件是否存在
import torch   # [新增] 用于保存和加载深度学习模型权重

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
        加载 YAML 配置，动态修改时间和股票池，然后根据日期决定是否训练
        """
        print(f"-> [Strategy] 正在加载配置文件: {yaml_path}")
        
        yaml = YAML(typ="safe", pure=True)
        config_file = Path(yaml_path).absolute().resolve()
        task_config = yaml.load(config_file.open(encoding='utf-8'))
        
        dataset_config = task_config["task"]["dataset"]
        model_config = task_config["task"]["model"]
        
        # ========================================================
        # 【新增】：从 YAML 中动态提取 test 的起始和结束时间
        # ========================================================
        try:
            test_segment = dataset_config["kwargs"]["segments"]["test"]
            # 将 ruamel 解析出的日期对象转为 YYYYMMDD 格式的字符串
            start_str = pd.to_datetime(str(test_segment[0])).strftime('%Y%m%d')
            end_str = pd.to_datetime(str(test_segment[1])).strftime('%Y%m%d')
            # 将动态生成的文件名绑定到 self 上
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

        # ========================================================
        # 核心修改：根据是否是训练日 (周六) 来决定行为
        # ========================================================
        if is_training_day:
            print("-> [Strategy] 【周末任务】今天是周六，正在利用最新数据重新训练模型...")
            self.model.fit(dataset, save_path=self.weight_path)
            # 保存 PyTorch 模型权重
            print(f"-> [Strategy] 模型重新训练完成！最新权重已保存至: {self.weight_path}")
            
        else:
            print("-> [Strategy] 【工作日任务】今天是交易日，跳过训练流程...")
            # 检查本地是否已经有训练好的权重文件
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
        
        # 兼容性处理：如果是单列 Series (旧模型)
        if isinstance(predictions, pd.Series):
            pred_df = predictions.to_frame(name='pred_center_return')
        else:
            pred_df = predictions.copy()
            # 如果是 FactorVAE 返回的多列 DataFrame，
            # 将代表确定性收益的 'pred_mu' 重命名为 'pred_center_return' 喂给下游网格策略
            if 'pred_mu' in pred_df.columns:
                pred_df = pred_df.rename(columns={'pred_mu': 'pred_center_return'})
            else:
                pred_df.columns = ['pred_center_return']
                
        # 动态获取文件名并导出 CSV
        export_name = getattr(self, 'csv_filename', 'default_predictions.csv')
        pred_df.to_csv(export_name)    
        print(f"-> [Strategy] 多维预测结果(包含mu, sigma, sample)已导出至: {export_name}")
        
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

def run_strategy(stock_list, today_str, current_prices, current_positions):
    """
    供外部调用的主函数接口
    """
    strategy = DeepGridStrategy()
    
    # 1. 解析日期，判断星期几 (0: 周一, 1: 周二 ... 5: 周六, 6: 周日)
    current_date = pd.to_datetime(today_str)
    day_of_week = current_date.dayofweek
    
    # 2. 设定周六 (5) 为训练日，其他时间为推理日
    is_training_day = (day_of_week == 5)
    
    start_date = (current_date - pd.Timedelta(days=60)).strftime('%Y%m%d')
    
    # 构建数据并根据是否是训练日来决定模型的行为
    dataset = strategy.build_dataset_and_model(
        stock_list=stock_list, 
        start_date=start_date, 
        end_date=today_str, 
        is_training_day=is_training_day
    )
    
    # 模型推理与动作生成
    predictions = strategy.get_model_prediction(dataset)
    actions = strategy.generate_actions(predictions, current_prices, current_positions)
    
    return actions

if __name__ == '__main__':
    import json
    import traceback

    print("=== 开始本地独立调试 strategy_model ===")
    
    test_stock_list = ['SZ000001', 'SH600050']
    
    # ==========================================
    # 调试测试指南：
    # 2023-10-25 是周三 -> 会触发【工作日极速推理】加载权重 (如果没有权重则紧急训练)
    # 2023-10-28 是周六 -> 会触发【周末重新训练】覆盖权重
    # ==========================================
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
        print(actions)
        print(json.dumps(actions, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"\n[-] [调试报错] 策略运行失败: {str(e)}")
        traceback.print_exc()
        
    print("\n=== 本地调试结束 ===")