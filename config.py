# config.py

# --- 数据与选股参数 ---
# BASE_SECTOR = ''      # 基础股票池
TOP_N_STOCKS = 50           # 最终保留的活跃股数量
LOOKBACK_DAYS = 50          # 评估活跃度的回溯天数

# --- 时间范围设定 ---
RESEARCH_START_DATE = '20100101'
RESEARCH_END_DATE = '20260313'

# --- 路径配置 ---

MODEL_NAME='factorvae'
# --- 训练相关 ---
QLIB_DIR="./qlib_data/my_custom_cn_data"
#QLIB_DIR="./qlib-main/qlib_data/cn_data"
WEIGHT_PATH=f"{MODEL_NAME}_model_weights.pth"
YAML_PATH = f'./model_code/{MODEL_NAME}/{MODEL_NAME}_config.yaml'
RESULTS_DIR = './results/'
CSV_OUTPUT_DIR = './qlib_source_csvs'

FEATURE_DICT = {
    "close_norm": "$close / Mean($close, 15)",
    "volume_norm": "$volume / Mean($volume, 15)",
    "return_15d": "$close / Ref($close, 15) - 1",
    "volatility": "Std($close, 15)",
    "vwap_ratio": "$vwap / $close"
}
LABEL_DICT = {
    "target_vwap": "Mean(Ref($vwap, -1), 3)" 
}

# --- 接口相关 ---
MINI_QMT_PATH= r'D:\国金QMT交易端模拟\userdata_mini'
ACCOUNT_ID="86004893"
# --- 远程 AI 服务器配置 (Linux) ---
LINUX_SERVER_IP = "192.168.1.154"
LINUX_SERVER_PORT = 1214
API_ENDPOINT = f"http://{LINUX_SERVER_IP}:{LINUX_SERVER_PORT}/get_target_actions"
# --- 交易相关 ---
MAX_CASH_PER_STOCK = 10000