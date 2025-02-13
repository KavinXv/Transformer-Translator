from typing import Optional  # 用于类型注解
import torch
import torch.nn as nn
import torch.nn.functional as F

# 一个非常大的负数，用于掩码操作
MY_INF = 1e12


# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        # 确保d_model是偶数
        # d_model: 词维度
        assert d_model % 2 == 0

        # 生成词位置编码和词维度编码

        # 生成一个从 0 到 max_len-1 的序列，表示每个词的位置索引
        # 例如：如果 max_seq_len = 10，则 i_seq = [0, 1, 2, ..., 9]
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len) # 位置序列

        # 生成一个从 0 到 d_model-2 的序列，表示每个位置的维度索引
        # 例如：如果 d_model = 512，则 i_dim = [0, 1, 2, ..., 256]
        j_seq = torch.linspace(0, d_model - 2, d_model // 2) # 维度序列

        # 生成一个网格矩阵，矩阵的行数是 max_seq_len，矩阵的列数是 d_model // 2
        # 矩阵的每一行是一个位置的位置编码
        # pos: 词的位置  two_i: 词的维度
        # 用于生成两个矩阵（或张量），这两个矩阵是通过将输入的两个一维张量（i_seq 和 j_seq）进行网格化扩展得到的
        # 它用于将位置索引和维度索引组合起来
        # pos shape: [max_seq_len, d_model // 2]
        # two_i shape: [max_seq_len, d_model // 2] 都是矩阵
        pos, two_i = torch.meshgrid(i_seq, j_seq)

        # 计算正弦和余弦位置编码
        # pe_2i shape: [max_seq_len, d_model // 2]
        # pe_2i_1 shape: [max_seq_len, d_model // 2]
        pe_2i = torch.sin(pos / 10000**(two_i / d_model)) # 偶数维度使用正弦
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model)) # 奇数维度使用余弦

        # 将偶数维度和奇数维度拼接在一起
        # pe shape: [max_seq_len, d_model]
        pe = torch.stack([pe_2i, pe_2i_1], dim=-1).reshape(1, max_seq_len, d_model)

        # 将位置编码注册为缓冲区，不参与参数更新
        self.register_buffer('pe', pe, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [n, seq_len, d_model]
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe  # 从位置编码中取出对应长度的位置编码
        assert seq_len <= pe.shape[1]  # 确保输入序列长度不超过最大长度
        assert d_model == pe.shape[2]  # 确保输入维度与位置编码维度一致

        # 对输入进行缩放
        rescaled_x = x * d_model**0.5
        # 将位置编码添加到输入张量中
        # 返回的张量形状与输入张量相同
        return rescaled_x + self.pe[:, 0:seq_len, :]  # 取出对应长度的位置编码




# 注意力机制函数
def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
    '''
    掩码的数据类型必须是 bool
    '''
    # q shape: [n, heads, q_len, d_k]   d_k: 词维度
    # k shape: [n, heads, k_len, d_k]
    # v shape: [n, heads, v_len, d_v]
    assert q.shape[-1] == k.shape[-1]  # 确保 q 和 k 的维度一致
    d_k = k.shape[-1]

    # 进行注意力计算
    # tmp shape: [n, heads, q_len, k_len]
    # 进行点积操作, 改k shape: [n, heads, d_k, k_len]
    # 这样与q shape: [n, heads, q_len, d_k]进行点积之后形状变成tmp shape
    tmp = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5

    # 如果提供了掩码，则将掩码的位置设置为那个非常大的负数
    if mask is not None:
        tmp.masked_fill_(mask, -MY_INF)

    # softmax 操作
    tmp = F.softmax(tmp, dim=-1)

    # 进行加权操作
    tmp = torch.matmul(tmp, v)
    return tmp



# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % heads == 0  # 确保 d_model 可以被 heads 整除\
        self.d_k = d_model // heads  # 计算每个 head 的维度
        self.heads = heads
        self.d_model = d_model

        # 初始化 Q、K、V 矩阵
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model) # 输出变换矩阵
        self.dropout = nn.Dropout(dropout) # Dropout 操作

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 确保 batch size 一致
        assert q.shape[0] == k.shape[0] == v.shape[0]

        # 确保 kv 序列长度一致
        assert k.shape[1] == v.shape[1]

        n, q_len = q.shape[0:2]
        n, k_len = k.shape[0:2]

        # 将输入进行线性变换
        # 在这里变成多头注意力机制
        # 形状: (n, heads, q_len, d_k) 为了后续计算方便
        # n: batch_size
        q_ = self.q(q).reshape(n, q_len, self.heads, self.d_k).transpose(1, 2)
        k_ = self.k(k).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)
        v_ = self.v(v).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)

        # 进行注意力计算
        attention_res = attention(q_, k_, v_, mask)

        # 将多头注意力的结果进行拼接并重塑形状
        # 形状: (n, q_len, d_model)
        contact_res = attention_res.transpose(1, 2).reshape(n, q_len, self.d_model)
        contact_res = self.out(contact_res) # Dropout 操作

        # 输出线性变换
        output = self.dropout(contact_res)
        return output

        

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)  # 第一层线性变换
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        self.layer2 = nn.Linear(d_ff, d_model)  # 第二层线性变换

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.dropout(F.relu(x))  # 应用 ReLU 和 Dropout
        x = self.layer2(x)
        return x
    

# 编码器层类
class EncoderLayer(nn.Module):
    def __init__(self, heads: int, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super().__init__()
        # 自注意力机制
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        # 前馈神经网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        # LayerNorm 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout 操作
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 多头注意力机制
        res = self.self_attention(x, x, x, src_mask)
        # 添加残差连接
        x = self.norm1(x + self.dropout1(res))
        # 前馈神经网络
        res = self.ffn(x)
        # 添加残差连接
        x = self.norm2(x + self.dropout2(res))
        return x
    

# 解码器层类
class DecoderLayer(nn.Module):
    def __init__(self, 
                 heads: int, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super().__init__()
        # 自注意力机制
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        # 编码器-解码器注意力机制
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        # 前馈神经网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        # LayerNorm 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout 操作
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                encoder_kv: torch.Tensor,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力机制
        tmp = self.self_attention(x, x, x, dst_mask) # 先对自己的输入进行注意力机制
        x = self.norm1(x + self.dropout1(tmp)) # 添加残差连接和归一化

        # 编码器-解码器注意力机制
        tmp = self.attention(x, encoder_kv, encoder_kv, src_dst_mask) # 对编码器的输出进行注意力机制
        x = self.norm2(x + self.dropout2(tmp)) # 添加残差连接和归一化

        # 先对自己的输入进行注意力机制，再将这个结果与编码器的输出进行注意力机制
        # 可以知道在预测下一个词时，更关注编码器输出中的哪一部分
        '''
        二、具体例子：翻译 “The cat sat on the mat” → “猫坐在垫子上”
        假设已生成前两个词 ["猫", "坐"]，现在要生成第三个词 “在”：

        步骤 1：掩码自注意力（处理目标序列自身）
        输入：["猫", "坐"]（已生成部分）。

        过程：

        模型通过自注意力分析这两个词的关系（例如“猫”是主语，“坐”是动作）。

        输出一个表示向量，编码了“猫坐”的上下文信息（比如动作发生的主体和动作本身）。

        步骤 2：编码器-解码器注意力（关联源序列）
        Query：来自步骤1的输出（表示“猫坐”的向量）。

        Key/Value：编码器输出的源序列表示（["The", "cat", "sat", "on", "the", "mat"]）。

        计算注意力权重：

        模型计算 Query 与每个 Key 的相似度，确定在生成“在”时需要关注哪些源词。

        权重结果示例：最高权重分配给 "on"，表示“在”对应英文的介词“on”。

        输出：加权后的 Value 向量（主要包含“on”的信息）。

        步骤 3：预测下一个词
        将步骤2的输出输入前馈网络，预测下一个词为“在”。
        '''
        
        # 前馈神经网络
        tmp = self.ffn(x)
        x = self.norm3(x + self.dropout3(tmp)) # 添加残差连接和归一化
        return x
    


# 编码器类
class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)   
        # 位置编码层
        self.pe = PositionalEncoding(d_model, max_seq_len)
        # 编码器层
        self.layers = nn.ModuleList(
            [EncoderLayer(heads, d_model, d_ff, dropout) for _ in range(n_layers)])
        # Dropout 操作
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 词嵌入
        x = self.embedding(x)
        # 位置编码
        x = self.pe(x)
        # Dropout 操作
        x = self.dropout(x)
        # 编码器层
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    


# 解码器类
class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # 位置编码层
        self.pe = PositionalEncoding(d_model, max_seq_len)
        # 解码器层
        self.layers = nn.ModuleList(
            [DecoderLayer(heads, d_model, d_ff, dropout) for _ in range(n_layers)])
        # Dropout 操作
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                encoder_kv: torch.Tensor,  
                dst_mask: Optional[torch.Tensor] = None, 
                src_dst_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 词嵌入
        x = self.embedding(x)
        # 位置编码
        x = self.pe(x)
        # Dropout 操作
        x = self.dropout(x)
        # 解码器层
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_dst_mask)

        return x

# Transformer 类
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 pad_idx: int,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 n_layers: int = 6,
                 heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        # 编码器
        self.encoder = Encoder(src_vocab_size, pad_idx, d_model, d_ff, n_layers, heads, dropout, max_seq_len)
        # 解码器
        self.decoder = Decoder(dst_vocab_size, pad_idx, d_model, d_ff, n_layers, heads, dropout, max_seq_len)
        # 输出层
        self.output = nn.Linear(d_model, dst_vocab_size)
        # 填充符的索引
        self.pad_idx = pad_idx

    # 生成掩码
    def generate_mask(self,
                      q_pad: torch.Tensor,
                      k_pad: torch.Tensor,
                      with_left_mask: bool) -> torch.Tensor:
        # q_pad shape: [n, q_len]
        # k_pad shape: [n, k_len]
        # q_pad k_pad dtype: bool
        assert q_pad.device == k_pad.device
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        # 生成掩码
        mask_shape = (n, 1, q_len, k_len)
        if with_left_mask:
            # 生成左下三角掩码
            mask = 1 - torch.tril(torch.ones(mask_shape))
        else:
            # 生成填充符掩码
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)

        # 填充掩码
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask
    
    # 前向传播
    def forward(self,
                src: torch.Tensor,
                dst: torch.Tensor) -> torch.Tensor:
        # src shape: [n, src_len]
        # dst shape: [n, dst_len]

        # 生成源序列掩码
        src_pad = src == self.pad_idx
        src_mask = self.generate_mask(src_pad, src_pad, False)

        # 生成目标序列掩码
        dst_pad = dst == self.pad_idx
        dst_mask = self.generate_mask(dst_pad, dst_pad, True)

        # 生成源-目标序列掩码
        src_dst_mask = self.generate_mask(dst_pad, src_pad, False)

        # 编码器输出
        encoder_kv = self.encoder(src, src_mask)

        # 解码器输出
        decoder_output = self.decoder(dst, encoder_kv, dst_mask, src_dst_mask)

        # 输出层
        output = self.output(decoder_output)
        return output



