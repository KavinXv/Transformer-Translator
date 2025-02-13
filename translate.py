import torch

# 导入需要的函数和模块
from data_load import (idx_to_sentence, load_cn_vocab,
                        load_en_vocab, maxlen)  # 导入词汇表加载函数和句子转换函数
from model import Transformer  # 导入Transformer模型定义

# 配置参数
batch_size = 1  # 设置每批次的大小为1
lr = 0.0001  # 学习率
d_model = 512  # 模型的维度
d_ff = 2048  # 前馈网络的维度
n_layers = 6  # Transformer层数
heads = 8  # 多头注意力机制的头数
dropout_rate = 0.2  # dropout比率
n_epochs = 60  # 训练的周期数

PAD_ID = 0  # 填充符号的ID，通常在序列填充时用到

def main():
    # 设置设备为GPU
    device = 'cuda'

    # 加载中文和英文的词汇表
    cn2idx, idx2cn = load_cn_vocab()  # 中文词汇表：从中文词汇到索引、从索引到中文词汇
    en2idx, idx2en = load_en_vocab()  # 英文词汇表：从英文词汇到索引、从索引到英文词汇

    # 初始化Transformer模型，传入目标和源语言的词汇表大小以及其他超参数
    model = Transformer(len(en2idx), len(cn2idx), 0, d_model, d_ff, n_layers,
                        heads, dropout_rate, maxlen)
    model.to(device)  # 将模型移到GPU
    model.eval()  # 将模型设置为评估模式（此模式下会禁用dropout等）

    # 加载已保存的模型权重
    model_path = 'model_save/best_model.pth'
    checkpoint = torch.load(model_path)  # 加载模型文件
    model.load_state_dict(checkpoint['model_state_dict'])  # 从指定路径加载模型参数

    # 输入的一些英文句子
    my_input = ['we', 'should', 'protect', 'environment']
    my_input = ['we', 'should', 'have', 'a', 'good', 'time']
    my_input = ['personages', 'of', 'the', 'beijing', 'political', 'circles', 'think', 'remark', 'in', 'essence',
                 'was', 'a', 'warning', 'for', 'the', 'taiwan', 'authorities', 'in', 'connection', 'with', 'their', 'recent', 'advocacy', 'of', 
                  'military']

    # 将输入句子转换为索引形式，并转换为PyTorch张量
    x_batch = torch.LongTensor([[en2idx[x] for x in my_input]]).to(device)

    # 将输入句子转换为英文句子的字符串形式（仅用于打印）
    cn_sentence = idx_to_sentence(x_batch[0], idx2en, True)
    print(cn_sentence)  # 打印英文句子

    # 初始化输出的目标句子，开始时全是PAD_ID
    y_input = torch.ones(batch_size, maxlen,
                         dtype=torch.long).to(device) * PAD_ID  # y_input用来存放输出的目标句子
    y_input[0] = en2idx['<S>']  # 输出句子的开始符号，通常是<START>

    # 使用模型进行翻译，逐步预测目标句子的下一个词
    with torch.no_grad():  # 禁用梯度计算，提高推理效率
        for i in range(1, y_input.shape[1]):  # 从第二个词开始生成（第一个词是<START>）
            y_hat = model(x_batch, y_input)  # 获取模型对目标句子的预测
            for j in range(batch_size):  # 对每个batch内的句子进行处理
                y_input[j, i] = torch.argmax(y_hat[j, i - 1])  # 选择当前时刻的词的最大概率

    # 将预测的目标句子转换为中文句子的字符串形式
    output_sentence = idx_to_sentence(y_input[0], idx2cn, True)
    print(output_sentence)  # 打印翻译结果（中文）

# 程序入口，调用main函数
if __name__ == '__main__':
    main()
