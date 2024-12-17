# By BUBBLE
# 2024-12-17
'''
ref : BEATs: Audio Pre-Training with Acoustic Tokenizers
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from BEATs import BEATs, BEATsConfig

# 配置日志
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainArguments:
    """
    Arguments related to training.
    """
    data_dir: str = field(default='../ESC-50-master/audio')
    csv_file: str = field(default='../ESC-50-master/meta/esc50.csv')
    batch_size: int = field(default=16)
    learning_rate: float = field(default=1e-4)
    num_epochs: int = field(default=20) # 差不多10轮就可以
    pretrained_path: str = field(default="../ckpt/BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt2.pt")
    output_model_path: str = field(default="../ckpt/finetuned_model.pt")
    train_mode: bool = field(default=False)

@dataclass
class ModelArguments:
    """
    Arguments related to model initialization and configuration.
    """
    checkpoint_path: str = field(default='../ckpt/finetuned_model_1.pt')

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, csv_file, mode, transform=None):
        """
        ESC-50 数据集加载与预处理
        Args:
            data_dir: 音频数据目录
            csv_file: 标签 CSV 文件路径
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 读取标签文件
        self.data = []

        with open(csv_file, "r") as f:
            lines = f.readlines()[1:]  # 跳过标题行
            for line in lines:
                filename, fold, target = line.strip().split(",")[:3]
                # 5 折作为测试集
                if (fold == "5" and mode == "test") or (fold != "5" and mode == "train"):
                    self.data.append((filename, int(target)))
        
    def __getitem__(self, idx):
        filename, label = self.data[idx]
        waveform, sr = torchaudio.load(os.path.join(self.data_dir, filename))
        # 转为 16kHz
        resampler = T.Resample(sr, 16000) #[1 , 8w] 5s的音频
        waveform = resampler(waveform)
        return waveform.squeeze(0), label
    
    def __len__(self):
        return len(self.data)

def train(model, train_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for waveforms, labels in progress_bar:
            waveforms = waveforms.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            # 获取 logits， 目前的策略是直接用交叉熵，并且没有实现论文中finetune阶段的随机mask
            logits, _ = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader):.4f}")

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", unit='batch')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = model(inputs)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

def main():
    # 参数解析
    parser = HfArgumentParser((TrainArguments, ModelArguments))
    train_args, model_args = parser.parse_args_into_dataclasses()

    # 加载数据
    train_dataset = ESC50Dataset(train_args.data_dir, train_args.csv_file, "train")
    train_loader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)

    test_dataset = ESC50Dataset(train_args.data_dir, train_args.csv_file, "test")
    test_loader = DataLoader(test_dataset, batch_size=train_args.batch_size, shuffle=False)

    # 设备配置 and 实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(train_args.pretrained_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    
    # 训练或评估
    if train_args.train_mode:
        # 加载模型参数
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.predictor = nn.Linear(cfg.encoder_embed_dim, 50)
        BEATs_model = BEATs_model.to(device)
        optimizer = optim.Adam(BEATs_model.parameters(), lr=train_args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        train(BEATs_model, train_loader, optimizer, criterion, device, num_epochs=train_args.num_epochs)
        torch.save(BEATs_model.state_dict(), train_args.output_model_path)
    else:
        BEATs_model.predictor = nn.Linear(cfg.encoder_embed_dim, 50)
        BEATs_model.load_state_dict(torch.load(model_args.checkpoint_path))
        BEATs_model = BEATs_model.to(device)
        evaluate(BEATs_model, test_loader, device)

if __name__ == "__main__":
    main()

