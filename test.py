# -*- coding: utf-8 -*-
import argparse
import cPickle
import random
import torch
import time
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from model import MwAN
from utils import padding
from preprocess.preprocess import filter_comms
from retrieval_elastic_eng_comm import elasticsearchVisitor

parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')
parser.add_argument('--model', type=str, default='model/model_best.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--vocab', type=str, default='data/word2id.obj',
                    help='path to save the final model')
args = parser.parse_args()

with open(args.model, 'rb') as f:
    if args.cuda:
        model = torch.load(f, map_location=lambda storage, loc: storage.cuda())
    else:
        model = torch.load(f, map_location=lambda storage, loc: storage)

with open(args.vocab, 'rb') as f:
    word2id = cPickle.load(f)

def map_sent_to_id(sent):
    output = []
    for word in sent:
        if word in word2id:
            output.append(word2id[word])
        else:
            # UNK token
            output.append(1)
    return output

def jaccard_similarity(s1, s2):
    """
    计算两个句子的雅可比相似度
    :param s1: 
    :param s2: 
    :return: 
    """
    vectorizer = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = vectorizer.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator

def test():
    vES = elasticsearchVisitor()
    request_amount = 10
    model.eval()
    with torch.no_grad():
        while True:
            raw_post = raw_input('input:')
            if raw_post == 'exit':
                break
            # post规范化
            raw_post = ' '.join(word_tokenize(raw_post))
            raw_post = ''.join([w.lower() for w in raw_post])
            # 选出90条候选评论
            candi_comms = vES.visitSpaceQuery(raw_post, request_amount*3)
            # comms规范化
            candi_comms = filter_comms(candi_comms[::-1], request_amount*3, 30)
            post = map_sent_to_id(raw_post.split())
            posts, _ = padding([post for _ in range(request_amount*3)], max_len=30)
            posts = torch.LongTensor(posts)
            if args.cuda:
                posts = posts.cuda()
            comms = list()
            for comm in candi_comms:
                comm = word_tokenize(comm)
                comm = map_sent_to_id(comm)
                comms.append(comm)
            comms, _ = padding([comm for comm in comms], max_len=30)
            comms = torch.LongTensor(comms)
            if args.cuda:
                comms = comms.cuda()
            start = time.time()
            output = model([posts, comms, False])
            print("model cost time %f"%(time.time() - start))
            print('all comm candidates:')
            score_comm = list()
            for i, comm in enumerate(candi_comms):
                jaccard = jaccard_similarity(raw_post, comm)
                score_comm.append([output[i, 0].item(), jaccard, comm])
            score_comm = sorted(score_comm, key = lambda x: x[0], reverse=True)
            for score, jaccard, comm in score_comm:
                print(str(score) + '\t' + str(jaccard) + '\t' + comm)
            print("------------------------------")
            print('match comms:')
            for score, jaccard, comm in score_comm:
                if score > 0.9 and jaccard < 0.4:
                    print(comm)

if __name__ == '__main__':
    test()
            