# -*- coding: utf-8 -*-
import re
from nltk.tokenize import word_tokenize

# 判断是否为英文文本并除去非法字符
def is_english(sent):
    en_cnt = 0
    all_cnt = 0
    words = sent.strip().split()
    filter_words = list()
    for w in words:
        en_flag = 1
        for c in w:
            if ord(c) > 126 and ord(c) != 8217:
                en_flag = 0
                break
        if en_flag:  
            en_cnt += len(w)
            # 单词小写化
            filter_words.append(w.lower())
        all_cnt += len(w)
    filter_sent = ' '.join(filter_words)
    if en_cnt >= int(all_cnt/2):
        return True, filter_sent
    else:
        return False, None

def filter_user(sent):
    words = sent.strip().split()
    filter_words = list()
    for w in words:
        if w[0] == '@' and len(w) > 1:
            continue
        filter_words.append(w)
    filter_sent = ' '.join(filter_words)
    return filter_sent

def filter_tag(sent):
    if len(sent) == 0:
        return sent
    words = sent.strip().split()
    filter_words = list()
    for w in words:
        if w[0] == '#' and len(w) > 1:
            filter_words.append(w[1:])
        else:
            filter_words.append(w)
    filter_sent = ' '.join(filter_words)
    return filter_sent

def filter_url(sent):
    url_flag = False
    words = sent.strip().split()
    filter_words = list()
    for w in words:
        r = re.search(r"(((ht|f)tp(s?)\:\/\/[0-9a-zA-Z])|(w{3}))([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%\$#_]*)?", w)
        if r is None:
            if "twitter.com" in w:
                url_flag = True
            else:
                filter_words.append(w)
        else:
            pos = r.span()
            # url在token的后半部分
            if pos[0] != 0:
                filter_words.append(w[0:pos[0]])
            url_flag = True
    filter_sent = ' '.join(filter_words)
    return url_flag, filter_sent

# 除去省略号
def filter_ellipsis(sent):
    if len(sent) == 0:
        return sent
    words = sent.strip().split()
    if words[-1][-1] == "…":
        return ' '.join(words[:-1])
    else:
        return sent       

def tokenize(sent):
    if len(sent) == 0:
        return sent
    words = word_tokenize(sent)
    tokenize_sent = ' '.join(words)
    return tokenize_sent

# 筛查检索出的dismatch comm
def filter_comms(sents, sent_amount, sent_len_limit):
    # 检索出的comm去重
    sents_set = set()
    for sent in sents:
        sents_set.add(sent)
    sents = list(sents_set)
    filter_sents = list()
    # 倒序筛查comm
    for sent in sents[::-1]:
        sent = filter_user(sent)
        url_flag, sent = filter_url(sent)
        if url_flag and len(sent.split()) < 3:
            continue
        is_en, sent = is_english(sent)
        if not is_en:
            continue
        sent = filter_tag(sent)
        sent = filter_ellipsis(sent)
        sent = tokenize(sent)
        # 去除句子最后只剩标点符号的情况，并保证comm至少长度为5
        if 5 <= len(sent.split()) <= sent_len_limit:
            none_en = True
            for w in sent:
                if 65<=ord(w)<=90 or 97<=ord(w)<=122:
                    none_en = False
                    break
            # 行内没有任何英文字符需要去除
            if not none_en:
                filter_sents.append(sent)
                if len(filter_sents) == sent_amount:
                    break
    return filter_sents      

def filter_pair(pair):
    post = list()
    comm = list()
    for line in pair:
        if line[:5] == "post:":
            post.append(line[6:])
        else:
            if line[:5] == "comm:":
                comm.append(line[6:])
            else:
                if len(comm) == 0:
                    post.append(line)
                else:
                    comm.append(line)

    filter_post = list()
    url_flag = False
    post_cnt = 0
    for line in post:
        line = filter_user(line)
        flag, line = filter_url(line)
        if flag:
            url_flag = True
        is_en, line = is_english(line)
        if not is_en:
            continue
        line = filter_tag(line)
        line = filter_ellipsis(line)
        line = tokenize(line)
        # 去除句子最后只剩标点符号的情况
        if len(line.split()) != 0:
            none_en = True
            for w in line:
                if 65<=ord(w)<=90 or 97<=ord(w)<=122:
                    none_en = False
                    break
            # 行内没有任何英文字符需要去除
            if not none_en:
                filter_post.append(line)
                post_cnt += len(line.split())
    # 整段都是非英文组成
    if len(filter_post) == 0:
        return False
    # 包含url且post过短
    if url_flag and post_cnt < 3:
        return False
    
    filter_comm = list()
    url_flag = False
    comm_cnt = 0
    for line in comm:
        line = filter_user(line)
        flag, line = filter_url(line)
        if flag:
            url_flag = True
        is_en, line = is_english(line)
        if not is_en:
            continue
        line = filter_tag(line)
        line = filter_ellipsis(line)
        line = tokenize(line)
        if len(line.split()) != 0:
            none_en = True
            for w in line:
                if 65<=ord(w)<=90 or 97<=ord(w)<=122:
                    none_en = False
                    break
            # 行内没有任何英文字符需要去除
            if not none_en:
                filter_comm.append(line)
                comm_cnt += len(line.split())
    # 整段都是非英文组成
    if len(filter_comm) == 0:
        return False
    # 包含url且post过短
    if url_flag and comm_cnt < 3:
        return False
    
    new_pair = "post: " + " \\n ".join(filter_post) + '\n' + "comm: " + " \\n ".join(filter_comm) + '\n' + "--------------------------------------------\n"
    return new_pair

def preprocess_pairs(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        fw = open('pre_english_comm_two.txt', 'a')
        fw_1 = open('useless_pairs.txt', 'a')
        pair = list()
        pair_cnt = 0
        pairs = str()
        useless_cnt = 0
        useless_pairs = str()
        for line in f.readlines():
            line = line.strip()
            # 过滤空行
            if len(line) == 0:
                continue
            if line == "--------------------------------------------":
                new_pair = filter_pair(pair)
                if new_pair:
                    pairs += new_pair
                    pair_cnt += 1
                    if pair_cnt % 100000 == 0:
                        fw.write(pairs)
                        print("pair:", pair_cnt)
                        pairs = str()
                else:
                    useless_pairs += '\n'.join(pair) + "\n--------------------------------------------\n"
                    useless_cnt += 1
                    if useless_cnt % 100000 == 0:
                        fw_1.write(useless_pairs)
                        print("useless:", useless_cnt)
                        useless_pairs == str()
                pair = list()
            else:
                pair.append(line)
        if pair_cnt % 100000 != 0:
            fw.write(pairs)
            print("pair:", pair_cnt)
        if useless_cnt % 100000 != 0:
            fw_1.write(useless_pairs)
            print("useless:", useless_cnt)
        fw.close()
        fw_1.close()


if __name__ == '__main__':
    file_name = 'english_comm_two.txt'
    preprocess_pairs(file_name)



