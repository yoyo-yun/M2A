# ======================================= #
# ----------- Data4BertModel ------------ #
# ======================================= #
import os
from M2A.dataset import SentenceProcessor
import pandas as pd
from collections import Counter


class IMDB(SentenceProcessor):
    NAME = 'IMDB'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.train.txt.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.dev.txt.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'imdb', 'imdb.test.txt.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test, train])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev, self.d_test)  # tuple(attributes) rather tuple(users, products)


class YELP_13(SentenceProcessor):
    NAME = 'YELP_13'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        super().__init__()
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.train.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.dev.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp_13', 'yelp-2013-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test, train])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test, train])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev, self.d_test)


class YELP_14(SentenceProcessor):
    NAME = 'YELP_14'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.train.ss'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.dev.ss'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp_14', 'yelp-2014-seg-20-20.test.ss'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test, train])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class MTL(SentenceProcessor):
    NAME = 'MTL'
    NUM_CLASSES = 2
    domain_list = ["apparel", "baby", "books", "camera_photo",
                   "dvd", "electronics", "health_personal_care", "imdb",
                   "kitchen_housewares", "magazines", "MR", "music",
                   "software", "sports_outdoors", "toys_games", "video"]
    # domain_list = ["apparel"]

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file_(os.path.join(data_dir, 'MTL'), "train")
        self.d_dev = self._read_file_(os.path.join(data_dir, 'MTL'), "dev")
        self.d_test = self._read_file_(os.path.join(data_dir, 'MTL'), "test")
        self.d_unlabel = self._read_unlabel_file_(os.path.join(data_dir, 'MTL'), "unlabel")

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        unlabel = self._create_examples(self.d_unlabel, 'unlabel')
        return tuple([train, dev, test, unlabel])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev, self.d_test)  # tuple(attributes) rather tuple(users, products)

    def _get_attributes(self, *datasets):
        users = Counter()
        products = Counter()
        for domain in self.domain_list:
            users.update([domain])
            products.update([domain])
        return tuple([users, products])

    def _read_file_(self, dataset, datatype):
        documents = []
        for domain in self.domain_list:
            datapath = os.path.join(dataset, "{}.task.{}".format(domain, datatype))
            pd_reader = pd.read_csv(datapath, header=None, skiprows=0, encoding="utf-8", sep='\t', engine='python')
            for i in range(len(pd_reader[0])):
                # u, i, t, l
                document = list([domain, domain, pd_reader[1][i], int(pd_reader[0][i]) + 1])
                documents.append(document)
        return documents

    def _read_unlabel_file_(self, dataset, datatype):
        documents = []
        for domain in self.domain_list:
            datapath = os.path.join(dataset, "{}.task.{}".format(domain, datatype))
            pd_reader = open(datapath, encoding="utf-8")
            for i in pd_reader.read().splitlines():
                text = i.rstrip()
                if len(text) == 0:
                    continue
                document = list([domain, domain, text, None])
                documents.append(document)
        return documents