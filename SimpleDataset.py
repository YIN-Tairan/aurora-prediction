import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class OMNISequenceDataset(Dataset):
    def __init__(self, parquet_file, input_days=3, predict_hours=3, stride_mins=60):
        """
        :param parquet_file: 处理后的 parquet 文件路径
        :param input_days: 输入观察窗口大小 (默认 3 天)
        :param predict_hours: 预测目标窗口大小 (默认 3 小时)
        :param stride_mins: 切割样本的滑动步长
        """
        print(f"正在加载 {parquet_file} 至内存...")
        self.df = pd.read_parquet(parquet_file)
        
        self.input_mins = int(input_days * 24 * 60)         # 4320 mins
        self.predict_mins = int(predict_hours * 60)         # 180 mins
        self.window_size = self.input_mins + self.predict_mins # 4500 mins
        self.stride = stride_mins
        
        self.segments = []      
        self.index_map = []     
        
        feature_cols = [col for col in self.df.columns if col != 'Segment_ID']
        grouped = self.df.groupby('Segment_ID')
        
        for seg_id, group in grouped:
            # 填补插值后可能残存的边界 NaN
            features_array = group[feature_cols].fillna(method='ffill').fillna(0).values
            tensor_data = torch.tensor(features_array, dtype=torch.float32)
            seg_length = len(tensor_data)
            
            # 只有长度大于 4500 的连续段才有资格被切分
            if seg_length >= self.window_size:
                self.segments.append(tensor_data)
                seg_idx = len(self.segments) - 1  
                
                for start_pos in range(0, seg_length - self.window_size + 1, self.stride):
                    self.index_map.append((seg_idx, start_pos))
                    
        print(f"Dataset 就绪！输入: {self.input_mins}min -> 预测: {self.predict_mins}min")
        print(f"可用样本总数: {len(self.index_map):,} 个")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seg_idx, start_pos = self.index_map[idx]
        
        # 提取整个 4500 分钟的长序列
        full_window = self.segments[seg_idx][start_pos : start_pos + self.window_size]
        
        # 将其切分为 X (前 4320 分钟) 和 y (后 180 分钟)
        X = full_window[:self.input_mins]
        y = full_window[self.input_mins:]
        
        # 备注：默认 y 包含了所有 37 个特征的未来走向。
        # 如果你只想预测特定特征 (如 BZ 或 AE-index)，可以这样切片：
        # y = full_window[self.input_mins:, target_feature_index]
        
        return X, y

# ================= 测试运行 =================
if __name__ == "__main__":
    dataset = OMNISequenceDataset(
        parquet_file='omni_ready_for_pytorch.parquet', 
        input_days=3, 
        predict_hours=3, 
        stride_mins=60
    )
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    
    for X_batch, y_batch in dataloader:
        print("\nBatch 取样成功!")
        print("X shape (Input):", X_batch.shape)   # 期望: [64, 4320, 37]
        print("y shape (Target):", y_batch.shape)  # 期望: [64, 180, 37]
        break