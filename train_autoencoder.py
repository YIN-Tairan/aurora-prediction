import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 数据集定义 (专为 Autoencoder 修改)
# ==========================================
class OMNIAutoencoderDataset(Dataset):
    def __init__(self, parquet_file, input_days=3, stride_mins=60):
        print(f"正在加载 {parquet_file} 至内存...")
        self.df = pd.read_parquet(parquet_file)
        
        # 自编码器不需要预测未来的 Y，只需要输入 X
        self.window_size = int(input_days * 24 * 60)  # 4320 mins
        self.stride = stride_mins
        
        feature_cols = [col for col in self.df.columns if col != 'Segment_ID']
        self.num_features = len(feature_cols)
        
        print("正在进行特征标准化 (Mean=0, Std=1)...")
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[feature_cols] = self.df[feature_cols].ffill().fillna(0)
        
        self.scaler = StandardScaler()
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        
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
        # 对于自编码器，输入 X 和目标 Y 是同一个东西
        X = self.segments[seg_idx][start_pos : start_pos + self.window_size]
        return X, X

# ==========================================
# 2. 1D-CNN 时间序列自编码器网络结构
# ==========================================
class Conv1DAutoencoder(nn.Module):
    def __init__(self, seq_len=4320, num_features=37, latent_dim=128):
        super(Conv1DAutoencoder, self).__init__()
        
        # -------------------
        # Encoder (编码器)
        # -------------------
        # 输入维度: [Batch, Channels(37), Length(4320)]
        self.encoder_conv = nn.Sequential(
            # 4320 -> MaxPool(3) -> 1440
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
            
            # 1440 -> MaxPool(3) -> 480
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
            
            # 480 -> MaxPool(4) -> 120
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4)
        )
        
        # 将压缩后的时间序列压扁，进入真正的隐空间 (Latent Space)
        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 120, 1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim) # 这里就是你要的 Latent Vector！
        )

        # -------------------
        # Decoder (解码器)
        # -------------------
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 256 * 120),
            nn.GELU()
        )
        
        self.decoder_conv = nn.Sequential(
            # 120 -> Upsample(4) -> 480
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            # 480 -> Upsample(3) -> 1440
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # 1440 -> Upsample(3) -> 4320
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Conv1d(in_channels=64, out_channels=num_features, kernel_size=5, padding=2)
            # 最后一层不需要激活函数，因为我们的目标是还原标准正态分布的数值
        )

    def forward(self, x):
        # PyTorch 的 Conv1d 需要的维度是 [Batch, Channels, Length]
        # 而我们的数据集输出的是 [Batch, Length, Channels]
        x = x.permute(0, 2, 1)  # 转换维度: [Batch, 37, 4320]
        
        # 编码
        encoded_features = self.encoder_conv(x)
        latent_vector = self.encoder_linear(encoded_features)
        
        # 解码
        decoded_features = self.decoder_linear(latent_vector)
        # Reshape 恢复到卷积层需要的形状 [Batch, 256, 120]
        decoded_features = decoded_features.view(-1, 256, 120) 
        
        reconstructed = self.decoder_conv(decoded_features)
        
        # 变回原始的输入维度 [Batch, Length, Channels]
        reconstructed = reconstructed.permute(0, 2, 1)
        
        return reconstructed

    def get_latent_embedding(self, x):
        """这是一个辅助工具函数，用于训练完成后直接提取数据的隐向量"""
        x = x.permute(0, 2, 1)
        encoded = self.encoder_conv(x)
        latent = self.encoder_linear(encoded)
        return latent

# ==========================================
# 3. 训练主逻辑
# ==========================================
def main():
    EPOCHS = 15
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    LATENT_DIM = 128   # 隐空间的维度大小，可以根据需要调整 (64, 128, 256)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {DEVICE}")

    dataset = OMNIAutoencoderDataset(
        parquet_file='omni_ready_for_pytorch.parquet', 
        input_days=3, 
        stride_mins=10
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
    
    model = Conv1DAutoencoder(seq_len=4320, num_features=dataset.num_features, latent_dim=LATENT_DIM).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    print("\n" + "="*50)
    print(f"🚀 开始训练自编码器 (隐空间维度: {LATENT_DIM})...")
    print("="*50)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", leave=False)
        
        for X_batch, _ in train_pbar:
            X_batch = X_batch.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # 前向传播：让网络重构自己
                reconstructed = model(X_batch)
                loss = criterion(reconstructed, X_batch)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_pbar.set_postfix({'recon_loss': f"{loss.item():.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        model.eval()
        test_loss = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Test ", leave=False)
        
        with torch.no_grad():
            for X_test, _ in test_pbar:
                X_test = X_test.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    reconstructed = model(X_test)
                    loss = criterion(reconstructed, X_test)
                test_loss += loss.item()
                test_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | 耗时: {epoch_time:.1f}s | Train MSE: {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f}")

    print("\n🎉 自编码器训练完成！")
    
    # 保存训练好的模型权重
    torch.save(model.state_dict(), "omni_autoencoder.pth")
    print("模型已保存为 omni_autoencoder.pth")

if __name__ == "__main__":
    main()