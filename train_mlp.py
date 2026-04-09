import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler # <--- 新增：标准化工具

# ==========================================
# 1. 数据集定义 (加入归一化)
# ==========================================
class OMNISequenceDataset(Dataset):
    def __init__(self, parquet_file, input_days=3, predict_hours=3, stride_mins=60):
        print(f"正在加载 {parquet_file} 至内存...")
        self.df = pd.read_parquet(parquet_file)
        
        self.input_mins = int(input_days * 24 * 60)         
        self.predict_mins = int(predict_hours * 60)         
        self.window_size = self.input_mins + self.predict_mins 
        self.stride = stride_mins
        
        # Exclude Segment_ID and columns named '4' through '12'
        exclude_cols = {'Segment_ID'} | {str(i) for i in range(3, 10)}  # '3' to '9' inclusive
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.num_features = len(feature_cols)
        print(f"特征列数 (不含 Segment_ID 和 '3'-'9'): {self.num_features}")
        
        # --- 核心修复：特征归一化 (Standardization) ---
        print("正在进行特征标准化 (Mean=0, Std=1)... 解决 NaN 问题")
        # 1. 替换极其罕见的无穷大值为 NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 2. 用 0 填补全部的 NaN
        self.df[feature_cols] = self.df[feature_cols].ffill().fillna(0)
        
        # 3. 实例化并拟合 StandardScaler
        self.scaler = StandardScaler()
        # 注意：这里我们对整个 DataFrame 进行了全局归一化
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        # ----------------------------------------------
        
        self.segments = []      
        self.index_map = []     
        
        grouped = self.df.groupby('Segment_ID')
        for seg_id, group in grouped:
            features_array = group[feature_cols].values
            tensor_data = torch.tensor(features_array, dtype=torch.float32)
            seg_length = len(tensor_data)
            
            if seg_length >= self.window_size:
                self.segments.append(tensor_data)
                seg_idx = len(self.segments) - 1  
                
                for start_pos in range(0, seg_length - self.window_size + 1, self.stride):
                    self.index_map.append((seg_idx, start_pos))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seg_idx, start_pos = self.index_map[idx]
        full_window = self.segments[seg_idx][start_pos : start_pos + self.window_size]
        X = full_window[:self.input_mins]
        y = full_window[self.input_mins:]
        return X, y

# ==========================================
# 2. 定义简单的多层感知机 (MLP)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, seq_len, num_features, pred_len):
        super(SimpleMLP, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        
        in_dim = int(seq_len * num_features)
        out_dim = int(pred_len * num_features)
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.pred_len, self.num_features)

# ==========================================
# 3. 训练与测试主逻辑
# ==========================================
def main():
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的计算设备: {DEVICE}")

    dataset = OMNISequenceDataset(
        parquet_file='omni_ready_for_pytorch.parquet', 
        input_days=3, 
        predict_hours=3, 
        stride_mins=60
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
    
    # 动态获取 seq_len
    model = SimpleMLP(seq_len=dataset.input_mins, num_features=dataset.num_features, pred_len=dataset.predict_mins).to(DEVICE)
    criterion = nn.MSELoss()
    # 加入了 weight_decay (L2正则化) 增加稳定性
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    print("\n" + "="*50)
    print("🚀 开始训练...")
    print("="*50)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", leave=False)
        
        for X_batch, y_batch in train_pbar:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
            scaler.scale(loss).backward()
            
            # --- 额外保险：梯度裁剪，防止偶尔的突变导致 NaN ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        model.eval()
        test_loss = 0.0
        test_mae = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Test ", leave=False)
        
        with torch.no_grad():
            for X_test, y_test in test_pbar:
                X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    preds = model(X_test)
                    loss = criterion(preds, y_test)
                    mae = torch.mean(torch.abs(preds - y_test))
                    
                test_loss += loss.item()
                test_mae += mae.item()
                test_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mae = test_mae / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | 耗时: {epoch_time:.1f}s | Train MSE: {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f} | Scaled MAE: {avg_test_mae:.4f}")

    print("\n🎉 测试结束，训练完毕！")

if __name__ == "__main__":
    main()