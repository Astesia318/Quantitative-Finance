import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# 1. 读取并合并数据
# ==========================================
print("-> 正在加载数据...")
df_pred = pd.read_csv("./results/factorvae_predictions_20210101_20260310.csv") 
df_real = pd.read_csv("./results/real_market_data_20210101_20260313.csv") # 替换为你真实数据的文件名

df = pd.merge(df_pred, df_real, on=['datetime', 'instrument'], how='inner')
df['datetime'] = pd.to_datetime(df['datetime'])

# ==========================================
# 2. 挑选股票和时间段
# ==========================================
target_stock = 'SH601216'  # 替换为你想要观察的股票，如 'SH600150'

if target_stock not in df['instrument'].values:
    target_stock = df['instrument'].iloc[0]
    print(f"-> 未找到指定股票，自动选择: {target_stock}")

df_stock = df[df['instrument'] == target_stock].sort_values('datetime')
# 截取几个月的细节来看，太长了会挤在一起
df_stock = df_stock[(df_stock['datetime'] >= '2021-01-01') & (df_stock['datetime'] <= '2022-01-01')]

# ==========================================
# 3. 绘制终极全景图表 (Sample + Mu + Sigma + Real)
# ==========================================
print(f"-> 正在绘制 {target_stock} 的终极对比图...")

fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

# (1) 绘制真实的日收益率 (蓝色实线带圆点)
ax.plot(df_stock['datetime'], df_stock['real_return'], 
        label='Real Return (True Market)', color='steelblue', alpha=0.9, linewidth=2, marker='o', markersize=4)

# (2) 绘制 FactorVAE 的随机采样收益率 (橙色线带 X 标记)
# ax.plot(df_stock['datetime'], df_stock['pred_sample'], 
#         label='Predicted Sample (Stochastic Draw)', color='darkorange', alpha=0.8, linewidth=1.5, marker='x', markersize=4)

# (3) 绘制预测的期望均值 (绿色虚线)
ax.plot(df_stock['datetime'], df_stock['pred_center_return'], 
        label='Expected Mean (Mu)', color='forestgreen', linestyle='--', alpha=0.8, linewidth=2)

# (4) 🌟 新增：利用 pred_sigma 绘制 1 倍标准差的“风险置信带”
upper_bound = df_stock['pred_center_return'] + df_stock['pred_sigma']
lower_bound = df_stock['pred_center_return'] - df_stock['pred_sigma']

# fill_between 填充上下界，选用浅绿色与 Mu 的深绿色呼应
# ax.fill_between(df_stock['datetime'], lower_bound, upper_bound, 
#                 color='mediumseagreen', alpha=0.2, label='Risk Band (Mu ± 1 Sigma)')

# ==========================================
# 4. 图表美化设置
# ==========================================
ax.set_title(f'FactorVAE Full View: Sample, Mean, Variance vs Real Return - {target_stock}', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Return Rate', fontsize=12)

# 添加 0 轴参考线 (涨跌分界线)
ax.axhline(0, color='black', linewidth=1, linestyle='-')

# 设置 X 轴日期格式
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

ax.legend(loc='upper right', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()