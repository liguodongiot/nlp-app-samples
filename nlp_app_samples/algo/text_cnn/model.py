import torch.nn as nn
import torch


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 windows_size, max_len,
                 feature_size, n_class, dropout=0.4):
        super(TextCNN, self).__init__()

        # embedding层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # 卷积层特征提取
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=max_len - h + 1),
                          ) for h in windows_size
        ])

        # 全连接层
        self.fc = nn.Linear(feature_size * len(windows_size), n_class)
        # dropout防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        x = [conv(x) for conv in self.conv1]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1))  # [batch, feature_size*len(windows_size)]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, n_class]
        return x


if __name__ == '__main__':
    x = torch.ones(32, 128)
    x = x.type(torch.LongTensor)
    model = TextCNN()
    x = model(x)
