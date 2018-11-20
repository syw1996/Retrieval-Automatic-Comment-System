# -*- coding: utf-8 -*-
import os
import cPickle
import json
import jieba

# 加载数据只需修改此函数即可
def train_data(path):
    print 'start process ', path
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            post = line[0].split()
            comm_true = line[1].split()
            comm_false = line[2].split()
            if len(post) > 30 or len(comm_true) > 30 or len(comm_false) > 30:
                print(len(post), len(comm_true), len(comm_false))
            if len(post) == 0 or len(comm_true) == 0 or len(comm_false) == 0:
                print(len(post), len(comm_true), len(comm_false))
            data.append([post, comm_true, comm_false])
    return data

def dev_data(path):
    print 'start process ', path
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if line == "--------------------------------------------\n":
                continue
            if line[:4] == "post":
                post = line.strip().split()[1:]
            elif line[:4] == "comm":
                comm = line.strip().split()[1:]
                data.append([post, comm, "1"])
            elif line[:8] == "mismatch":
                mis_comm = line.strip().split()[1:]
                data.append([post, mis_comm, "0"])
    return data

def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:3]]
    print 'word type size ', len(wordCount)-2
    return wordCount


def build_word2id(wordCount, threshold=10):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    if os.path.exists("data/vocab_50000.txt"):
        with open("data/vocab_50000.txt", 'r') as fr:
            for word in fr.readlines():
                word2id[word.strip()] = len(word2id)
    else:
        for word in wordCount:
            if wordCount[word] >= threshold:
                if word not in word2id:
                    word2id[word] = len(word2id)
            # 将词拆成字对英文不适用
            '''
            else:
                chars = list(word)
                for char in chars:
                    if char not in word2id:
                        word2id[char] = len(word2id)
            '''
    print 'processed word size ', len(word2id)
    return word2id


def transform_train_to_id(raw_data, word2id):
    data = []

    def map_word_to_id(word):
        output = []
        if word in word2id:
            output.append(word2id[word])
        # 将词拆成字对英文不适用
        else:
            '''
            chars = list(word)
            for char in chars:
                if char in word2id:
                    output.append(word2id[char])
                else:
            '''
            # UNK token
            output.append(1)
        return output

    def map_sent_to_id(sent):
        output = []
        for word in sent:
            output.extend(map_word_to_id(word))
        return output

    for one in raw_data:
        post = map_sent_to_id(one[0])
        comm1 = map_sent_to_id(one[1])
        comm2 = map_sent_to_id(one[2])
        data.append([post, comm1, comm2])
    return data

def transform_dev_to_id(raw_data, word2id):
    data = []

    def map_word_to_id(word):
        output = []
        if word in word2id:
            output.append(word2id[word])
        # 将词拆成字对英文不适用
        else:
            '''
            chars = list(word)
            for char in chars:
                if char in word2id:
                    output.append(word2id[char])
                else:
            '''
            # UNK token
            output.append(1)
        return output

    def map_sent_to_id(sent):
        output = []
        for word in sent:
            output.extend(map_word_to_id(word))
        return output

    for one in raw_data:
        post = map_sent_to_id(one[0])
        comm = map_sent_to_id(one[1])
        flag = one[2]
        data.append([post, comm, flag])
    return data


def process_data(data_path, threshold):
    train_file_path = data_path + 'train.txt'
    dev_file_path = data_path + 'dev.txt'
    test_file_path = data_path + 'test.txt'
    # test_a_file_path = data_path + 'ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    # test_b_file_path = data_path + 'ai_challenger_oqmrc_testb_20180816/ai_challenger_oqmrc_testb.json'
    path_lst = [train_file_path, dev_file_path, test_file_path]
    output_path = [data_path + x for x in ['train.pickle', 'dev.pickle', 'test.pickle']]
    return _process_data(path_lst, threshold, output_path)


def _process_data(path_lst, word_min_count=5, output_file_path=[]):
    raw_data = []
    train_file_path = path_lst[0]
    raw_data.append(train_data(train_file_path))
    dev_file_path = path_lst[1]
    raw_data.append(dev_data(dev_file_path))
    word_count = build_word_count([y for x in raw_data for y in x])
    test_file_path = path_lst[2]
    raw_data.append(dev_data(test_file_path))
    with open('data/word-count.obj', 'wb') as f:
        cPickle.dump(word_count, f)
    word2id = build_word2id(word_count, word_min_count)
    with open('data/word2id.obj', 'wb') as f:
        cPickle.dump(word2id, f)
    i = 0
    for one_raw_data, one_output_file_path in zip(raw_data, output_file_path):
        with open(one_output_file_path, 'wb') as f:
            if i == 0:
                one_data = transform_train_to_id(one_raw_data, word2id)
            else:
                one_data = transform_dev_to_id(one_raw_data, word2id)
            i += 1
            cPickle.dump(one_data, f)
    return len(word2id)

def build_word_embedding():
    with open('/home/sunyawei/glove.840B.300d.txt', 'r') as fr:
        glove_word_emb = dict()
        for line in fr.readlines():
            word = line.strip().split()[0]
            glove_word_emb[word] = line
    with open('data/vocab_50002.txt', 'r') as fr:
        word_embedding = str()
        for line in fr.readlines():
            word = line.strip()
            if word in glove_word_emb:
                word_embedding += glove_word_emb[word]
            else:
                emb = [word] + ['0']*300
                word_embedding += ' '.join(emb) + '\n'          
    with open('data/word_emb_50002.txt', 'w') as fw:
        fw.write(word_embedding)

if __name__ == '__main__':
    process_data('data/', 5)
    # build_word_embedding()
