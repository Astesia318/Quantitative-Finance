import qlib
from qlib.data import D
import config
import pandas as pd

print("-> 正在初始化 Qlib 引擎...")
# 确保这里的路径与你 config.py 或 lstm_config.yaml 中的 provider_uri 一致
qlib.init(provider_uri=config.QLIB_DIR, region="cn")

# 设定与你测试集完全一致的时间范围和股票池
start_time = '2021-01-01'
end_time = '2026-03-13'
market = 'all'

print(f"-> 正在从底层数据库提取 {market} 的真实行情数据...")

# ==========================================
# 定义我们要提取的真实数据指标 (完美对齐你的模型)
# ==========================================
fields = [
    # 1. 真实未来收益率 (对齐你 yaml 中的 label 标准答案)
    # 计算公式：(后天收盘价 / 明天收盘价) - 1
    "Ref($close, -2) / Ref($close, -1) - 1",
    
    # 2. 真实历史波动率 (作为 pred_sigma 的对比基准)
    # 计算公式：过去 20 个交易日“真实日收益率”的标准差
    "Std($close / Ref($close, 1) - 1, 20)"
]

names = [
    "real_return", 
    "real_sigma"
]

# 批量拉取数据 (Qlib 会自动处理前复权和停牌对齐)
instruments = D.instruments(market)
df_real = D.features(instruments, fields, start_time, end_time)
df_real.columns = names

# 剔除由于上市不足 20 天或未来停牌导致的空值
df_real = df_real.dropna()

# 导出为与预测结果格式完全相同的 CSV
export_name = config.RESULTS_DIR+f"real_market_data_{start_time.replace('-','')}_{end_time.replace('-','')}.csv"
df_real.to_csv(export_name)

print(f"-> 提取完成！真实数据已导出至: {export_name}")
print("\n=== 数据预览 ===")
print(df_real.head(10))