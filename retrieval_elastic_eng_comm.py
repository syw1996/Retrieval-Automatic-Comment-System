#!/usr/bin/env python3
#_*_ coding:utf-8 _*_
from datetime import datetime
from elasticsearch import Elasticsearch
import time, sys


class elasticsearchVisitor(object):
    """
    访问elasticsearch的接口类
    """

    def __init__(self):
        self.es = Elasticsearch(
                # 服务器ip及端口(根据自己设置)
                'xxx.xxx.xxx.xxx:xxxx',
                # 用户(根据自己设置)
                http_auth=('user1', 'user2'),
                use_ssl=True,
                verify_certs=False,
                )
        self.stpFset = self._loadStpDic()


    def _loadStpDic(self):
        stpSet = set([])
        with open("./ext_stopword.dic") as f:
            for line in f:
                line = line.strip()
                stpSet.add(line)
                stpFset = frozenset(stpSet)
            return stpFset

    def uniqQuery(self, query):
        seg_query_list = map(
                lambda x: x,
                query.split())
        seg_query_set = set(seg_query_list)
        stp_seg_query_list = list(seg_query_set.difference(self.stpFset))
        stp_seg_query_list.sort(key=seg_query_list.index)
        return " ".join(stp_seg_query_list)

    def _cleanAnswer(self, text):
        result = []
        text = text.strip().lower()
        words = text.split()
        for each in words:
            if each not in self.stpFset:
                result.append(each)
        return ' '.join(result)

    def uniqAnswer(self, answers):
        result = set()
        uniq_pool = set()
        uniq_pool.add("")
        for i in answers:
            if self._cleanAnswer(i) not in uniq_pool:
                uniq_pool.add(self._cleanAnswer(i))
                result.add(i)
            else:
                pass
        return result

    def visitSpaceQuery(self, query, request_amount, err_amount=None):
        """
        使用空格作为分词器标志
        返回值：(answers)
        """
        answers = []
        search_title_res = []
        search_cmnt_res = []
        search_multi_res = []
        if request_amount < 100 and err_amount is not None:
            request_amount = 300
        #elif request_amount > 1000:
        #    request_amount = 3000
        else:
            request_amount = request_amount*3
        amount_title = int(request_amount*0.25)
        amount_cmnt = int(request_amount*0.25)
        amount_multi = int(request_amount-int(request_amount*0.25)*2)
        start_time = time.time()
        # 返回title中搜索的结果
        search_title = self.es.search(
                index="english_comment",
                doc_type='eng_comm',
                body={"query": {
                    "match": {
                        "post": {
                            "query": query,
                            "analyzer": "ignorecase",
                            "minimum_should_match": "40%"
                            }
                        }
                    },
                      "size": amount_title}
                )
        for i in search_title['hits']['hits']:
            search_title_res.append(i['_source']['comm'])
        title_time = time.time()
        print('search_title using time %s' % str(title_time-start_time))
        # 返回cmnt中搜索的结果
        search_cmnt = self.es.search(
                index='english_comment',
                doc_type='eng_comm',
                body={'query': {
                    'match': {
                        'comm': {
                            "query": query,
                            "analyzer": "ignorecase",
                            "minimum_should_match": "30%"
                            }
                        }
                    },
                      'size': amount_cmnt}
                )
        for i in search_cmnt['hits']['hits']:
            search_cmnt_res.append(i['_source']['comm'])
        cmnt_time = time.time()
        print('search_cmnt using time %s' % str(cmnt_time-title_time))
        # 返回title和cmnt两个域的cross_fields搜索的结果
        search_multi = self.es.search(
                index='english_comment',
                doc_type='eng_comm',
                body={'query': {
                    'multi_match': {
                        "query": query,
                        "type": "cross_fields",
                        "analyzer": "ignorecase",
                        "fields": ["post", "comm"],
                        "minimum_should_match": "20%",
                        "tie_breaker": 0.3,
                        }
                    },
                      'size': amount_multi}
                )
        for i in search_multi['hits']['hits']:
            search_multi_res.append(i['_source']['comm'])
        mult_time = time.time()
        print('search_multi using time %s' % str(mult_time-cmnt_time))
        if err_amount is None:
            listall = search_multi_res + search_title_res + search_cmnt_res
        else:
            if err_amount % 3 != 0:
                listall = search_multi_res[-int(err_amount/3):] + search_title_res[-int(err_amount/3):] + search_cmnt_res[-(err_amount - int(err_amount/3)*2):]
            else:
                listall = search_multi_res[-int(err_amount/3):] + search_title_res[-int(err_amount/3):] + search_cmnt_res[-int(err_amount/3):]
        #answers = set(listall)
        # time_mark1 = time.time()
        
        #answers = self.uniqAnswer(listall)
        answers = listall
        # time_mark2 = time.time()
        # print('%d ms'%(time_mark2 - time_mark1), file=sys.stderr)
        return answers


if __name__ == "__main__":
    # 连接elasticsearch，默认端口9200
    vES = elasticsearchVisitor()
    while 1:
        print("example: Have you showed him your Minecraft gaming videos?")
        query = input("input:")
        amount = int(input("amount:"))
        if amount < 100:
            print('当amount小于100时,默认返回100个结果,大于1000时,只返回3000个结果')
        #query = vES.uniqQuery(query)
        print('分词后: %s' % query)
        tmp = vES.visitSpaceQuery(query, amount, 10)
        for id,each in enumerate(tmp):
            print("%d: %s" % (id, each))
        print(len(tmp))
        print('='*50)
       # vES.pprint(tmp[0],tmp[1],tmp[2])
