# -*- coding: utf-8 -*-
import re
import argparse
import cPickle
import random
import torch
# from sklearn.metrics import classification_report

from model import MwAN
from twitter_preprocess import process_data
from utils import *

parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling ')

parser.add_argument('--data', type=str, default='data/',
                    help='location directory of the data corpus')
parser.add_argument('--threshold', type=int, default=5,
                    help='threshold count of the word')
parser.add_argument('--epoch', type=int, default=50,
                    help='training epochs')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='hidden size of the model')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=1000,
                    help='# of batches to see the training error')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model/',
                    help='path to save the final model')
parser.add_argument('--pretrain', type=str, default=None,
                    help='path to pretrain model')

args = parser.parse_args()

# 词表限定为词频最高前5万个(除了PAD和UNK)
vocab_size = 50002

with open(args.data + 'train.pickle', 'rb') as f:
    train_data = cPickle.load(f)

with open(args.data + 'test.pickle', 'rb') as f:
    dev_data = cPickle.load(f)
# dev_data = sorted(dev_data, key=lambda x: len(x[1]))

print('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))

with open(args.data + 'word_emb_50002.txt', 'rb') as f:
    print('load word embedding')
    word_embedding = torch.FloatTensor(vocab_size, args.emsize)
    for i,line in enumerate(f.readlines()):
        for j,d in enumerate(line.strip().split()[1:]):
            word_embedding[i,j] = float(d)
    if args.cuda:
        word_embedding = word_embedding.cuda()

if args.pretrain is None:
    print('initiate model')
    model = MwAN(vocab_size=vocab_size, embedding_size=args.emsize, encoder_size=args.nhid, word_embedding=word_embedding, drop_out=args.dropout)
    start_epoch = 0
else:
    print('load pretrain model')
    pre_epoch = int(re.findall(r'\d+', args.pretrain)[0])
    start_epoch = pre_epoch + 1
    with open(args.pretrain, 'rb') as f:
        model = torch.load(f)

print('Model total parameters:', get_model_parameters(model))
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adamax(model.parameters())

def train(epoch):
    model.train()
    # data = shuffle_data(train_data, 1)
    total_loss = 0.0
    for num, i in enumerate(range(0, len(train_data), args.batch_size)):
        one = train_data[i:i + args.batch_size]
        if len(one) != args.batch_size:
            continue
        # len(x) = 4
        post, _ = padding([x[0] for x in one], max_len=30)
        comm1, _ = padding([x[1] for x in one], max_len=30)
        comm2, _ = padding([x[2] for x in one], max_len=30)
        # print post.shape, comm1.shape, comm2.shape
        y = torch.ones([args.batch_size, 1])
        post, comm1, comm2, y = torch.LongTensor(post), torch.LongTensor(comm1), torch.LongTensor(comm2), torch.FloatTensor(y)
        if args.cuda:
            post = post.cuda()
            comm1 = comm1.cuda()
            comm2 = comm2.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        # 随机输入全正确或者全错误
        if random.randint(0,1):
            loss = model([post, comm1, y, True])
        else:
            loss = model([post, comm2, 0 * y, True])
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if (num + 1) % args.log_interval == 0:
            print '|------epoch {:d} train loss is {:f}  eclipse {:.2f}%------|'.format(epoch,
                                                                                        total_loss / args.log_interval,
                                                                                        i * 100.0 / len(train_data))
            with open('model/loss.txt', 'a') as fw:
                fw.write('|------epoch {:d} train loss is {:f}  eclipse {:.2f}%------|\n'.format(epoch,
                                                                                        total_loss / args.log_interval,
                                                                                        i * 100.0 / len(train_data)))
            total_loss = 0


def test():
    model.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(dev_data), args.batch_size):
            one = dev_data[i:i + args.batch_size]
            if len(one) != args.batch_size:
                continue
            post, _ = padding([x[0] for x in one], max_len=30)
            comm, _ = padding([x[1] for x in one], max_len=30)
            y = [int(x[2]) for x in one]
            # print(post.shape, comm.shape)
            post, comm = torch.LongTensor(post), torch.LongTensor(comm)
            if args.cuda:
                post = post.cuda()
                comm = comm.cuda()
            output = model([post, comm, False])
            '''
            for j in range(args.batch_size):
                print(j+1, y[j], output[j,0].item())
            exit()
            
            
            if output.size() != ((args.batch_size,1)):
                continue
            '''
            for j in range(args.batch_size):
                if (output[j,0].item() > 0.5 and y[j] == 1) or (output[j,0].item() <= 0.5 and y[j] == 0):
                    r += 1
            a += len(one)
    return r * 100.0 / a


def main():
    best = 0.0
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        acc = test()
        if acc > best:
            best = acc
        with open(args.save + 'model_%d.pt'%epoch, 'wb') as f:
            torch.save(model, f)
        print 'epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best)
        with open('model/accuracy.txt', 'a') as fw:
            fw.write('epcoh {:d} dev acc is {:f}, best dev acc {:f}\n'.format(epoch, acc, best))


if __name__ == '__main__':
    main()
