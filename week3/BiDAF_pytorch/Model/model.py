import torch.nn as nn

from Model.layers import HighWay, Convolution, OutputLayer, ModelingLayer, ContextEmbedding
from Model.attention import BiDirectionalAttention


class Model(nn.Module):
    def __init__(self, embed_size, char_size, hidden_size, kernel_size, n_char, type, char_embed_size, dropout):
        super(Model, self).__init__()

        self.highway = HighWay(embed_size, char_size)
        self.char_embedding = Convolution(kernel_size, n_char + 1, char_embed_size, char_size, dropout)
        self.context_embedding = ContextEmbedding(hidden_size, dropout)

        self.attention = BiDirectionalAttention(hidden_size)

        self.modelinglayer = ModelingLayer(type, hidden_size, dropout)

        self.outputlayer = OutputLayer(hidden_size, dropout)

    def forward(self, sentence_context, sentence_question, char_sentence_question, char_sentence_context):
        # 1. char embedding层
        # 2. word embedding层
        # 3. context embedding层
        char_embeds_context = self.char_embedding(char_sentence_context)
        input_context = self.highway(sentence_context, char_embeds_context)

        H = self.context_embedding(input_context)

        char_embeds_question = self.char_embedding(char_sentence_question)
        input_question = self.highway(sentence_question, char_embeds_question)

        U = self.context_embedding(input_question)

        # 4. 注意力流层
        U_toggler, H_toggler = self.attention(H, U)
        # 5. 模型层
        G, M = self.modelinglayer(U_toggler, H_toggler, H)

        # 6. 输出层
        start, end = self.outputlayer(G, M)

        return start, end
