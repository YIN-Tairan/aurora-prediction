import pandas as pd
import numpy as np

# ==========================================
# 1. 核心超参数配置区
# ==========================================
INPUT_FILE = 'hro_data_sample.txt'
OUTPUT_PARQUET = 'omni_ready_for_pytorch.parquet'

MAX_MISSING_MINUTES = 15         
WINDOW_DAYS = 3.125  # 3天输入(72h) + 3小时预测(3h) = 75h = 3.125天
STRIDE_MINUTES = 10              

window_minutes = int(WINDOW_DAYS * 24 * 60) # 4500 分钟
max_time_diff = MAX_MISSING_MINUTES + 1  

# First, read the input file's header to resolve the actual column count and structure. The header looks like the following
"""
<HTML>
<HEAD><TITLE>OMNIWeb Results</TITLE></HEAD>
<BODY>
<center><font size=5 color=red>OMNIWeb Plus Browser Results </font></center><br>
<B>Listing for omni_min data from 2005010100 to 2005010211</B><hr><pre>Selected parameters:
 1 Year
 2 Day
 3 Hour
 4 Minute
 5 ID for IMF spacecraft
 6 ID for SW Plasma spacecraft
 7 # of points in IMF averages 
 8 # of points in Plasma averages
 9 Percent of Interpolation
10 Timeshift
11 RMS, Timeshift
12 RMS Min_var
13 Time btwn observations,sec
14 Field magnitude average, nT
15 BX, nT (GSE, GSM)
16 BY, nT (GSE)
17 BZ, nT (GSE)
18 BY, nT (GSM)
19 BZ, nT (GSM)
20 RMS SD B scalar, nT
21 RMS SD field vector, nT
22 Speed, km/s
23 Vx Velocity,km/s
24 Vy Velocity, km/s
25 Vz Velocity, km/s
26 Proton Density, n/cc
27 Proton Temperature, K
28 Flow pressure, nPa
29 Electric field, mV/m 
30 Plasma beta
31 Alfven mach number
32 S/C, Xgse,Re
33 S/C, Ygse,Re
34 S/c, Zgse,Re
35 BSN location, Xgse,Re
36 BSN location, Ygse,Re
37 BSN location, Zgse,Re
38 AE-index, nT
39 AL-index, nT
40 AU-index, nT
41 SYM/D, nT
42 SYM/H, nT
43 ASY/D, nT
44 ASY/H, nT
45 PCN-index
46 Magnetosonic Mach number
"""

header_feature_map = {}
with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    in_header_block = False
    for line in f:
        stripped = line.strip()
        if not in_header_block:
            if stripped.endswith('Selected parameters:'):
                in_header_block = True
            continue

        parts = stripped.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            break

        header_feature_map[int(parts[0])] = parts[1]

with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    title_tokens = None
    for line in f:
        stripped = line.strip()
        if stripped.startswith('YYYY '):
            title_tokens = stripped.split()
            break

if title_tokens is None:
    raise ValueError(f'Could not find the OMNI title line in {INPUT_FILE}.')

time_column_aliases = {'YYYY': 'Year', 'DOY': 'DOY', 'HR': 'Hour', 'MN': 'Minute'}
time_columns = [time_column_aliases[token] for token in title_tokens if token in time_column_aliases]
feature_columns = [int(token) for token in title_tokens if token.isdigit()]
spacecraft_id_columns = [
    col_num for col_num, label in header_feature_map.items()
    if label in {'ID for IMF spacecraft', 'ID for SW Plasma spacecraft'}
]

print("1/4 正在一次性读取并清洗全部数据...")
# 使用标题行动态解析出的列名，避免硬编码列数
col_names = time_columns + feature_columns
df = pd.read_csv(INPUT_FILE, sep=r'\s+', header=None, names=col_names, dtype=str, on_bad_lines='skip')

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
time_diffs = df['Datetime'].diff().dt.total_seconds().div(60.0).fillna(0.0)
is_large_gap = time_diffs > max_time_diff

spacecraft_gap_starts = pd.Series(False, index=df.index)
for col in spacecraft_id_columns:
    is_missing = pd.to_numeric(df[col], errors='coerce').eq(99)
    missing_groups = (~is_missing | is_large_gap).cumsum()
    missing_minutes = time_diffs.where(is_missing, 0.0).groupby(missing_groups, sort=False).cumsum()
    exceeded_limit = is_missing & (missing_minutes > max_time_diff)
    spacecraft_gap_starts |= exceeded_limit & ~exceeded_limit.shift(fill_value=False)

df['Chunk_ID'] = (is_large_gap | spacecraft_gap_starts).cumsum()

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
# 过滤掉与时间列重复的编号列，以及不应参与插值的航天器 ID 列
excluded_feature_labels = {'Year', 'Day', 'Hour', 'Minute', 'ID for IMF spacecraft', 'ID for SW Plasma spacecraft',
                           'Percent of Interpolation', 'Timeshift','RMS, Timeshift','RMS Min_var','Time btwn observations,sec'}
feature_cols = [col for col in feature_columns if header_feature_map.get(col) not in excluded_feature_labels]
missing_flags = ['99','999','9999', '99999', '999999', '9999999','99.9','99.99', '999.9', '999.99', '9999.99', '99999.9', '999999.', '9999999.']

# 统一转数值并替换 OMNI 占位符
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].replace([float(f) for f in missing_flags], np.nan)

print("4/4 执行高精度重采样与动态插值 (此步耗时较长，请稍候)...")
processed_chunks_list = []

for idx, row in valid_chunks.iterrows():
    mask = df['Chunk_ID'] == idx
    chunk_data = df.loc[mask, ['Datetime', *feature_cols]].copy()
    
    chunk_data.set_index('Datetime', inplace=True)
    
    # 1分钟重采样与插值
    chunk_data = chunk_data.resample('1min').mean()
    chunk_data = chunk_data.interpolate(method='linear', limit=MAX_MISSING_MINUTES)
    
    chunk_data['Segment_ID'] = idx
    processed_chunks_list.append(chunk_data)

final_dataset = pd.concat(processed_chunks_list)
final_dataset.columns = [str(c) for c in final_dataset.columns] 
final_dataset.to_parquet(OUTPUT_PARQUET)
print(f"\n✅ 数据集构建完毕！已保存至 {OUTPUT_PARQUET}")
