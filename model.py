# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class MwAN(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoder_size, word_embedding=None, drop_out=0.2):
        super(MwAN, self).__init__()
        self.drop_out=drop_out
        # self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_size)
        if word_embedding is not None:
            self.embedding.weight.data.copy_(word_embedding)
        self.p_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.c_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)

        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        self.gru_agg = nn.GRU(12 * encoder_size, encoder_size, batch_first=True, bidirectional=True)
        """
        prediction layer
        """
        self.Wp = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, 1, bias=False)

        self.criterion = nn.BCELoss()

        self.initiation()

    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def comm_forward(self, post, comm):
        p_embedding = self.embedding(post)
        c_embedding = self.embedding(comm)
        hp, _ = self.p_encoder(p_embedding)
        #print hp.size()
        hp=F.dropout(hp,self.drop_out)
        hc, _ = self.c_encoder(c_embedding)
        #print hc.size()
        hc=F.dropout(hc,self.drop_out)
        _s1 = self.Wc1(hp).unsqueeze(1)
        _s2 = self.Wc2(hc).unsqueeze(2)
        #print _s1.size(), _s2.size()
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        ptc = ait.bmm(hp)
        _s1 = self.Wb(hp).transpose(2, 1)
        sjt = hc.bmm(_s1)
        ait = F.softmax(sjt, 2)
        ptb = ait.bmm(hp)
        _s1 = hp.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptd = ait.bmm(hp)
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptm = ait.bmm(hp)
        _s1 = hc.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        pts = ait.bmm(hc)
        aggregation = torch.cat([hc, pts, ptc, ptd, ptb, ptm], 2)
        aggregation_representation, _ = self.gru_agg(aggregation)
        sj = self.vp(torch.tanh(self.Wp(hp))).transpose(2, 1)
        rp = F.softmax(sj, 2).bmm(hp)
        sj = F.softmax(self.vc(self.Wc1(aggregation_representation) + self.Wc2(rp)).transpose(2, 1), 2)
        rc = sj.bmm(aggregation_representation)
        # 归一化
        # encoder_output = F.sigmoid(self.prediction(rc))     
        score = F.sigmoid(self.prediction(rc.squeeze()))       
        return score
    
    def forward(self, inputs):
        if inputs[-1]:
            [post, comm, y, is_train] = inputs
        else:
            [post, comm, is_train] = inputs
        score = self.comm_forward(post, comm)
        if not is_train:
            return score
        loss = self.criterion(score, y)
        return loss


