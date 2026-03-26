from qlib.model.base import Model
import pandas as pd

class FileModel(Model):
    def __init__(self, pred_path):
        # 接收你在 YAML 中配置的预测结果路径
        self.pred_path = pred_path

    def fit(self, dataset):
        # 【关键】：什么都不做，直接跳过训练！
        print(f"Skipping training. Will load predictions from {self.pred_path}")
        pass 

    def predict(self, dataset):
        # 直接读取并返回事先准备好的预测分数
        # 假设你存的是 pickle 格式，且包含 datetime, instrument 的 MultiIndex
        pred_df = pd.read_pickle(self.pred_path)
        
        # 如果你的文件里是一个 DataFrame，通常返回它的一维 Series 格式
        if isinstance(pred_df, pd.DataFrame):
            return pred_df.iloc[:, 0]
        return pred_df