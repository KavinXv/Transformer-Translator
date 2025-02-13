# 导入必要的库
import codecs  # 用于处理文件编码
import os  # 用于处理文件路径和目录操作
import random  # 用于随机打乱数据

import numpy as np  # 用于数值计算
import regex  # 用于正则表达式操作
import requests  # 用于从网络下载数据

# 设置最小词频，低于此词频的词将被编码为 <UNK>
min_cnt = 0
# 设置句子的最大长度
maxlen = 50

# 定义训练和测试数据文件的路径
source_train = './data/cn.txt'
target_train = './data/en.txt'
source_test = './data/cn.test.txt'
target_test = './data/en.test.txt'


# 加载词汇表
def load_vocab(language):
    # 确保语言是中文或英文
    assert language in ['cn', 'en']
    # 从词汇表文件中读取词汇，并过滤掉词频低于 min_cnt 的词
    vocab = [
        line.split()[0] for line in codecs.open(
            'data/{}.txt.vocab.tsv'.format(language), 'r',
            'utf-8').read().splitlines() if int(line.split()[1]) >= min_cnt
    ]
    # 创建词到索引和索引到词的映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    # vocab = ["的", "是", "我", "你"]
    # word2idx = {"的": 0, "是": 1, "我": 2, "你": 3}
    # idx2word = {0: "的", 1: "是", 2: "我", 3: "你"}
    return word2idx, idx2word


# 加载中文词汇表
def load_cn_vocab():
    word2idx, idx2word = load_vocab('cn')
    return word2idx, idx2word


# 加载英文词汇表
def load_en_vocab():
    word2idx, idx2word = load_vocab('en')
    return word2idx, idx2word


# 创建训练或测试数据
def create_data(source_sents, target_sents):
    # 加载中文和英文词汇表
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    # 初始化列表，用于存储索引化的句子、源句子和目标句子
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # 将源句子和目标句子转换为索引序列，并在句子前后添加 <S> 和 </S> 标记
        x = [
            cn2idx.get(word, 1)   # 没找到就返回1，即<UNK>
            for word in ('<S> ' + source_sent + ' </S>').split()
        ]  # 1: OOV, </S>: End of Text
        y = [
            en2idx.get(word, 1)
            for word in ('<S> ' + target_sent + ' </S>').split()
        ]
        # 如果句子长度不超过最大长度，则将其加入列表
        if max(len(x), len(y)) <= maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # 对句子进行填充，使其长度一致
    X = np.zeros([len(x_list), maxlen], np.int32)
    Y = np.zeros([len(y_list), maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, maxlen - len(x)],
                          'constant',
                          constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, maxlen - len(y)],
                          'constant',
                          constant_values=(0, 0))

    return X, Y, Sources, Targets


# 加载数据
def load_data(data_type):
    # 根据数据类型选择源文件和目标文件
    if data_type == 'train':
        source, target = source_train, target_train
    elif data_type == 'test':
        source, target = source_test, target_test
    assert data_type in ['train', 'test']
    # 读取并清理源句子和目标句子
    cn_sents = [
        regex.sub("[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(source, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]
    en_sents = [
        regex.sub("[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(target, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]

    # 创建数据
    X, Y, Sources, Targets = create_data(cn_sents, en_sents)
    return X, Y, Sources, Targets


# 加载训练数据
def load_train_data():
    X, Y, _, _ = load_data('train')
    return X, Y


# 加载测试数据
def load_test_data():
    X, Y, _, _ = load_data('test')
    return X, Y


# 获取批量数据的索引
def get_batch_indices(total_length, batch_size):
    # 确保批量大小不超过数据总长度
    assert (batch_size <=
            total_length), ('Batch size is large than total data length.'
                            'Check your data or change batch size.')
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)  # 随机打乱索引
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index:current_index + batch_size], current_index


# 将索引序列转换为句子
def idx_to_sentence(arr, vocab, insert_space=False):
    res = ''
    first_word = True
    for id in arr:
        word = vocab[id.item()]

        if insert_space and not first_word:
            res += ' '
        first_word = False

        res += word

    return res


# 下载文件
def download(url, dir, name=None):
    os.makedirs(dir, exist_ok=True)  # 创建目录
    if name is None:
        name = url.split('/')[-1]  # 从 URL 中提取文件名
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Install {name} ...')
        open(path, 'wb').write(requests.get(url).content)  # 下载文件
        print('Install successfully.')


# 下载数据
def download_data():
    data_dir = 'dldemos/Transformer/data'
    urls = [('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/corpora/cn.txt'),
            ('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/corpora/en.txt'),
            ('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/preprocessed/cn.txt.vocab.tsv'),
            ('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/preprocessed/en.txt.vocab.tsv')]
    for url in urls:
        download(url, data_dir)  # 下载每个文件


# 主函数
if __name__ == '__main__':
    download_data()  # 下载数据