import torch
import torch.nn as nn
import torch.nn.functional as F


class HighWay(nn.Module):
    """
    高速神经网络
    """

    def __init__(self, word_dim, char_dim):
        """

        :param word_dim: 词向量维度
        :param char_dim: 字符向量维度
        """
        super(HighWay, self).__init__()

        self.Dense1 = nn.Linear(word_dim + char_dim, word_dim + char_dim)
        self.transform = nn.Linear(word_dim + char_dim, word_dim + char_dim)

    def forward(self, word, char):
        input = torch.cat([char, word], dim=2)

        output = F.relu_(self.Dense1(input))
        transform = torch.sigmoid_(self.transform(input))

        return output * transform + (1 - transform) * input


class Convolution(nn.Module):
    """
    卷积层
    """
    def __init__(self, kernel_size, char_size, embedding_dim, output_channels, dropout):
        super(Convolution, self).__init__()
        self.embedding = nn.Embedding(char_size, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, output_channels, kernel_size)

        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, input):
        input = self.embedding(input)
        input = input.transpose(1, 2)

        output1 = self.conv1(input)
        output1 = self.dropout1(output1)

        output1, _ = torch.max(output1, dim=2)
        output1 = output1.unsqueeze(0)
        output = self.relu(output1)

        return output

class ContextEmbedding(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ContextEmbedding, self).__init__()

        self.context_embedding = nn.LSTM(2 * hidden_size, hidden_size, num_layers=2, bidirectional=True,
                                         batch_first=True, dropout=dropout)

    def forward(self, input):
        # print(input.shape)
        output, (hidden_state, cell_state) = self.context_embedding(input, None)

        return output

class ModelingLayer(nn.Module):
    """
    模型层
    """
    def __init__(self, type, h_size, dropout):
        super(ModelingLayer, self).__init__()

        self.type = type
        self.m = nn.LSTM(h_size * 8, h_size, bidirectional=True, num_layers=2, batch_first=True,
                         dropout=dropout)

    def forward(self, U_toggler, H_toggler, H):
        if self.type == 'concat':
            G = torch.cat([H, U_toggler, H * U_toggler, H * H_toggler], dim=2)
            # 编码层输入是上层的输出G，再经过一次Bi - LSTM得到
            M, (hidden, cell_state) = self.m(G, None)

            return G, M
        else:
            raise NotImplementedError("Correct type has not been passed")




class OutputLayer(nn.Module):
    """
    输出层
    """
    def __init__(self, hidden_size, dropout):
        super(OutputLayer, self).__init__()

        self.start = nn.Linear(hidden_size * 10, 1)
        self.end = nn.Linear(hidden_size * 10, 1)

        self.m_to_m2 = nn.LSTM(2 * hidden_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True,
                               dropout=dropout)

    def forward(self, G, M):
        input1 = torch.cat([G, M], dim=2)

        input2, (hidden, cell_state) = self.m_to_m2(M, None)
        input2 = torch.cat([G, input2], dim=2)

        starts = self.start(input1).view(1, -1)
        ends = self.end(input2).view(1, -1)

        return starts, ends



