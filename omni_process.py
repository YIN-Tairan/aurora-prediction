import pandas as pd
import numpy as np

# ==========================================
# 1. 核心超参数配置区
# ==========================================
INPUT_FILE = 'omni_15.txt'
OUTPUT_PARQUET = 'omni_ready_for_pytorch.parquet'

MAX_MISSING_MINUTES = 30         
WINDOW_DAYS = 3.125  # 3天输入(72h) + 3小时预测(3h) = 75h = 3.125天
STRIDE_MINUTES = 10              

window_minutes = int(WINDOW_DAYS * 24 * 60) # 4500 分钟
max_time_diff = MAX_MISSING_MINUTES + 1  

print("1/4 正在一次性读取并清洗全部数据...")
# 强制指定 41 列（根据之前你提供的数据格式：4列时间 + 37列特征）
# 这样可以防止 Pandas 被首行的 <HTML> 误导而跳过真实的宽数据行
col_names = list(range(41))
df = pd.read_csv(INPUT_FILE, sep=r'\s+', header=None, names=col_names, dtype=str, on_bad_lines='skip')

# 重命名时间列，剩下的特征列依然是 4 到 40
df.rename(columns={0: 'Year', 1: 'DOY', 2: 'Hour', 3: 'Minute'}, inplace=True)

# 尝试转换时间列为数字，遇到文本（如 <HTML>）会变成 NaN
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['DOY'] = pd.to_numeric(df['DOY'], errors='coerce')
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
df['Minute'] = pd.to_numeric(df['Minute'], errors='coerce')

# 剔除所有时间列解析失败的行（完美过滤 HTML 表头和页脚）
df = df.dropna(subset=['Year', 'DOY', 'Hour', 'Minute']).copy()

# 规范化格式以拼接 Datetime
df['Year'] = df['Year'].astype(int).astype(str)
df['DOY'] = df['DOY'].astype(int).astype(str).str.zfill(3)
df['Hour'] = df['Hour'].astype(int).astype(str).str.zfill(2)
df['Minute'] = df['Minute'].astype(int).astype(str).str.zfill(2)

date_str = df['Year'] + df['DOY'] + df['Hour'] + df['Minute']
df['Datetime'] = pd.to_datetime(date_str, format='%Y%j%H%M')

# ==========================================
# 2. 连续有效数据段 (Chunks) 识别
# ==========================================
print("2/4 正在识别连续的有效数据段...")
time_diffs = df['Datetime'].diff().dt.total_seconds() / 60.0
is_large_gap = time_diffs > max_time_diff
df['Chunk_ID'] = is_large_gap.cumsum()

chunk_stats = df.groupby('Chunk_ID')['Datetime'].agg(['min', 'max'])
chunk_stats['Duration_Mins'] = (chunk_stats['max'] - chunk_stats['min']).dt.total_seconds() / 60.0
valid_chunks = chunk_stats[chunk_stats['Duration_Mins'] >= window_minutes].copy()

if valid_chunks.empty:
    print(f"\n[错误] 未找到长达 {WINDOW_DAYS} 天的连续数据段。")
    exit()
else:
    total_blocks = int(((valid_chunks['Duration_Mins'] - window_minutes) // STRIDE_MINUTES + 1).sum())
    print(f"  -> 共找到 {len(valid_chunks)} 个超长连续数据段。预计可切出 {total_blocks:,} 个样本。")

# ==========================================
# 3. 读取全量特征并执行插值保存
# ==========================================
print("\n3/4 开始替换缺失值占位符...")
# 提取所有的特征列（即除了时间相关的列）
feature_cols = [col for col in df.columns if col not in ['Datetime', 'Chunk_ID', 'Year', 'DOY', 'Hour', 'Minute']]
missing_flags = ['99.99', '999.9', '999.99', '9999.99', '99999.9', '999999.', '9999999.']

# 统一转数值并替换 OMNI 占位符
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].replace([float(f) for f in missing_flags], np.nan)

print("4/4 执行高精度重采样与动态插值 (此步耗时较长，请稍候)...")
processed_chunks_list = []

for idx, row in valid_chunks.iterrows():
    mask = df['Chunk_ID'] == idx
    chunk_data = df.loc[mask].copy()
    
    chunk_data.set_index('Datetime', inplace=True)
    # 丢弃不需要的辅助列，只保留真正的物理特征
    chunk_data.drop(columns=['Chunk_ID', 'Year', 'DOY', 'Hour', 'Minute'], inplace=True)
    
    # 1分钟重采样与插值
    chunk_data = chunk_data.resample('1min').mean()
    chunk_data = chunk_data.interpolate(method='linear', limit=MAX_MISSING_MINUTES)
    
    chunk_data['Segment_ID'] = idx
    processed_chunks_list.append(chunk_data)

final_dataset = pd.concat(processed_chunks_list)
final_dataset.columns = [str(c) for c in final_dataset.columns] 
final_dataset.to_parquet(OUTPUT_PARQUET)
print(f"\n✅ 数据集构建完毕！已保存至 {OUTPUT_PARQUET}")
