import os
import random

from retrieval_elastic_eng_comm import elasticsearchVisitor
from preprocess import filter_comms

MAX_LEN = 30
TRAIN_SIZE = 3000000
DEV_SIZE = 1000
TEST_SIZE = 1000
VOCAB_SIZE = 50000

def build_word_count():
    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1
    with open('pre_english_comm_deduplicated.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    wordCount = {}
    for line in lines:
        if line == "--------------------------------------------\n":
            continue
        sents = ''.join(line[6:].strip().split('\\n '))
        words = sents.split()
        add_count(words)

    print('word type size ', len(wordCount))
    return wordCount

def build_vocab():
    wordCount = build_word_count()
    wordCount = sorted(wordCount.items(), key = lambda x: x[1], reverse=True)
    vocab = list()
    for i in range(VOCAB_SIZE):
        vocab.append(wordCount[i][0])
    with open('vocab_%d.txt'%VOCAB_SIZE, 'w') as fw:
        fw.write('\n'.join(vocab))
    return set(vocab)

def get_train_dev_test():
    if os.path.exists('vocab_%d.txt'%VOCAB_SIZE):
        with open('vocab_%d.txt'%VOCAB_SIZE, 'r') as fr:
            vocab = set()
            for w in fr.readlines():
                vocab.add(w.strip())
    else:
        vocab = build_vocab()
    with open('pre_english_comm_deduplicated.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    all_comms = set()
    post_comm_dict = dict()
    for line in lines:
        if line == "--------------------------------------------\n":
            continue
        if line[:4] == "post":
            post = ''.join(line[6:].strip().split('\\n '))
            if len(post.split()) > MAX_LEN:
                post = None
        if line[:4] == "comm":
            comm = ''.join(line[6:].strip().split('\\n '))
            if len(comm.split()) <= MAX_LEN:
                all_comms.add(comm)
                if post is not None:
                    if post not in post_comm_dict:
                        post_comm_dict[post] = [comm]
                    else:
                        post_comm_dict[post].append(comm)
    all_comms = list(all_comms)
    comm_amount = len(all_comms)

    # 满足长度不超过30，有5个及以上comm，还能100%覆盖5万vocab的post
    multi_post = list()
    for p in post_comm_dict:
        # post和comm长度需要至少为5
        if len(p.split()) >= 5 and len(post_comm_dict[p]) >= 5:
            words = p.split()
            comm_cnt = 0
            for c in post_comm_dict[p]:
                if len(c.split()) >= 5:
                    comm_cnt += 1
                words.extend(c.split())
            if comm_cnt < 5:
                continue
            vocab_word_cnt = 0.0
            for w in words:
                if w in vocab:
                    vocab_word_cnt += 1
            if vocab_word_cnt == len(words):
                multi_post.append(p)
    print('match post num: ', len(multi_post))

    vES = elasticsearchVisitor()
    request_amount = 5000
    dev_test_post = set()
    
    if not os.path.exists("dev.txt"):
        dev_pairs = str()
        for i,p in enumerate(multi_post):
            dev_test_post.add(p)
            post = 'post: ' + p + '\n'
            comms = str()
            comm_cnt = 0
            for j,c in enumerate(post_comm_dict[p]):
                if len(c.split()) >= 5:
                    comms += 'comm%d: '%j + c + '\n'
                    comm_cnt += 1
                    if comm_cnt == 10:
                        break
            # mismatch_comms需要再经过预处理
            mismatch_comms = vES.visitSpaceQuery(p, request_amount, comm_cnt*60)
            mismatch_comms = filter_comms(mismatch_comms, comm_cnt*3, MAX_LEN)
            mis_comms = str()
            for j,mis_c in enumerate(mismatch_comms):
                mis_comms += 'mismatch%d: '%j + mis_c + '\n'
            dev_pairs += post + comms + mis_comms + '--------------------------------------------\n'
            if i == DEV_SIZE-1:
                break
        with open('dev.txt', 'w') as fw:
            fw.write(dev_pairs)
            print("dev bulid finish")
    else:
        with open("dev.txt", 'r') as fr:
            for line in fr.readlines():
                if line[0:4] == 'post':
                    dev_test_post.add(line[6:].strip())

    if not os.path.exists("test.txt"):    
        test_pairs = str()
        for i,p in enumerate(multi_post):
            if i < DEV_SIZE:
                continue
            dev_test_post.add(p)
            post = 'post: ' + p + '\n'
            comms = str()
            comm_cnt = 0
            for j,c in enumerate(post_comm_dict[p]):
                if len(c.split()) >= 5:
                    comms += 'comm%d: '%j + c + '\n'
                    comm_cnt += 1
                    if comm_cnt == 10:
                        break
            # mismatch_comms需要再经过预处理
            mismatch_comms = vES.visitSpaceQuery(p, request_amount, comm_cnt*60)
            mismatch_comms = filter_comms(mismatch_comms, comm_cnt*3, MAX_LEN)
            mis_comms = str()
            for j,mis_c in enumerate(mismatch_comms):
                mis_comms += 'mismatch%d: '%j + mis_c + '\n'
            test_pairs += post + comms + mis_comms + '--------------------------------------------\n'
            if i == TEST_SIZE + DEV_SIZE - 1:
                break
        with open('test.txt', 'w') as fw:
            fw.write(test_pairs)
            print("test bulid finish")
    else:
        with open("test.txt", 'r') as fr:
            for line in fr.readlines():
                if line[0:4] == 'post':
                    dev_test_post.add(line[6:].strip())

    f_train = open('train.txt', 'a')
    train_pairs = str()
    i = 0
    line_i = 0
    for line in lines:
        line_i += 1
        if line == "--------------------------------------------\n":
            continue
        if line[:4] == "post":
            post = ''.join(line[6:].strip().split('\\n '))
            if len(post.split()) > MAX_LEN or post in dev_test_post:
                post = None
        if line[:4] == "comm" and post != None:
            comm = ''.join(line[6:].strip().split('\\n '))
            if len(comm.split()) > MAX_LEN:
                continue
            while True:
                idx = random.randint(0, comm_amount-1)
                if comm != all_comms[idx]:
                    break
            train_pairs += post + '\t' + comm + '\t' + all_comms[idx] + '\n'
            i += 1
            if i % 100000 == 0:
                print(i)
                f_train.write(train_pairs)
                train_pairs = str()
                if i == 3000000:
                    break
    f_train.close()

if __name__ == '__main__':
    get_train_dev_test()





