import time
import torch
import torch.nn as nn
import logging
from data_load import (get_batch_indices, load_cn_vocab,
                       load_en_vocab, load_train_data,
                       maxlen)
from model import Transformer
import os
import time

# 配置参数
batch_size = 64  # 批量大小
lr = 0.0001  # 学习率
d_model = 512  # 模型维度
d_ff = 2048  # 前馈神经网络的隐藏层维度
n_layers = 6  # 编码器和解码器的层数
heads = 8  # 多头注意力的头数
dropout_rate = 0.2  # Dropout 概率
n_epochs = 100  # 训练轮数
PAD_ID = 0  # 填充符的索引
save_epoch = 20  # 每过 save_epoch 轮保存一次模型
log_file = 'training.log'  # 日志文件路径
best_model_filename = 'best_model.pth'  # 最高准确率模型保存路径
model_save_dir = './model_save'  # 模型保存目录
is_continue = 1  # 是否继续训练

# 创建模型保存目录
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# 配置日志
log_file = os.path.join(model_save_dir, log_file)
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# 主函数
def main():
    device = 'cuda'  # 使用 GPU 进行训练

    # 加载中文和英文的词汇表
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    # 加载训练数据
    Y, X = load_train_data()

    print_interval = 100  # 每隔 100 个批次打印一次训练信息

    # 初始化 Transformer 模型
    model = Transformer(len(en2idx), len(cn2idx), PAD_ID, d_model, d_ff,
                        n_layers, heads, dropout_rate, maxlen)
    model.to(device)  # 将模型移动到 GPU

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # 定义损失函数，忽略填充符的损失
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # 检查是否有已有的模型
    start_epoch = 0
    exist_model_path = os.path.join(model_save_dir, best_model_filename)  # 修改为使用文件名
    if os.path.exists(exist_model_path) and is_continue:
        checkpoint = torch.load(exist_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        logging.info(f'Resuming training from epoch {start_epoch} with best accuracy {best_acc}')
    else:
        best_acc = 0.0  # 如果没有找到已有模型，从头开始训练

    tic = time.time()  # 记录训练开始时间
    cnter = 0  # 计数器，用于记录训练批次
    t1 = time.time()  

    # 开始训练
    for epoch in range(start_epoch, n_epochs):
        model.train()  # 设置模型为训练模式
        epoch_loss = 0.0  # 记录每轮的总损失
        epoch_acc = 0.0  # 记录每轮的总准确率
        num_batches = 0  # 记录每轮的批次数量

        # 获取每个批次的索引
        for index, _ in get_batch_indices(len(X), batch_size):
            # 将数据转换为张量并移动到 GPU
            x_batch = torch.LongTensor(X[index]).to(device)  # 英文句子
            y_batch = torch.LongTensor(Y[index]).to(device)  # 中文句子

            # 将目标句子分为输入和标签
            y_input = y_batch[:, :-1]  # 去掉最后一个词作为输入
            y_label = y_batch[:, 1:]  # 去掉第一个词作为标签

            # 前向传播，计算模型输出
            y_hat = model(x_batch, y_input)

            # 计算准确率
            y_label_mask = y_label != PAD_ID  # 忽略填充符
            preds = torch.argmax(y_hat, -1)  # 获取预测结果
            correct = preds == y_label  # 计算预测正确的词
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)  # 计算准确率

            # 计算损失
            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))  # 重塑形状以计算损失
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = criterion(y_hat, y_label)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 梯度裁剪
            optimizer.step()  # 更新参数

            # 更新每轮的损失和准确率
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

        # 计算用时
        t2 = time.time()
        dt = t2 - t1
        t1 = t2

        # 计算每轮的平均损失和准确率
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        epoch_log_message = (f'Epoch {epoch + 1} , '
                             f'Avg Loss: {avg_loss}, Avg Acc: {avg_acc}, time: {dt:.2f}s')
        print(epoch_log_message)  # 打印到控制台
        logging.info(epoch_log_message)  # 写入日志文件

        # 每过 save_epoch 轮保存一次模型
        if (epoch + 1) % save_epoch == 0:
            model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, model_save_path)
            logging.info(f'Model saved to {model_save_path}')

        # 更新最高准确率并保存最佳模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_model_path = os.path.join(model_save_dir, best_model_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, best_model_path)
            logging.info(f'New best model saved with accuracy: {best_acc}')

    # 训练结束
    logging.info('Training completed.')


# 程序入口
if __name__ == '__main__':
    main()
