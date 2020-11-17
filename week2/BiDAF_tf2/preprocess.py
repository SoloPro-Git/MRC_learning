import numpy as np
import data_io as pio
word_maxlen=10
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

class Preprocessor:
    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.stride = stride
        self.charset = set()
        self.ch2id, self.id2ch = self.build_charset()
        self.word2id, self.id2word = {'[PAD]':0, '[CLS]':1, '[SEP]':2,'[UNK]':3},{}



    def glove_2_embedding(self,word2id_dict):
        """
        从文件中读取所有词的词向量
        按照给定的 词-idx 字典去获取embedding matrix
        :return:
        """
        embeddings_index = dict()
        with open('glove.6B.50d.txt',encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Loaded %s word vectors from Glove.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word2id_dict), 50))
        for word, i in word2id_dict.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_embedding_matrix(self):
        # TODO: 1) preprocess.py 实现加载glove到embedding matrix  （30'）
        self.embedding_matrix = self.glove_2_embedding(self.word2id)
        print(f'embedding_matrix shape is {self.embedding_matrix.shape} ')
        return self.embedding_matrix

    def build_charset(self):
        for fp in self.datasets_fp:
            self.charset |= self.dataset_info(fp)

        self.charset = sorted(list(self.charset))
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))
        self.ch2id = dict(zip(self.charset, idx))
        self.id2ch = dict(zip(idx, self.charset))
        print(self.ch2id, self.id2ch)
        return self.ch2id, self.id2ch

    def dataset_info(self, inn):
        charset = set()
        dataset = pio.load(inn)

        for _, context, question, answer, _ in self.iter_cqa(dataset):
            charset |= set(context) | set(question) | set(answer)
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen = max(self.max_clen, len(question))

        return charset

    def iter_cqa(self, dataset):
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text, answer_start

    def encode(self, context, question):
        question_encode_word, question_encode_char = self.convert2id(question, begin=True, end=True)
        left_length = self.max_length - len(question_encode_word)
        context_encode_word, context_encode_char = self.convert2id(context, maxlen=left_length, end=True)
        cq_encode_word = question_encode_word + context_encode_word
        cq_encode_char = question_encode_char + context_encode_char

        assert len(cq_encode_word) == self.max_length

        return cq_encode_word, cq_encode_char

    def word2char(self, sent, word_maxlen=None):
        """
        把每个词分隔成单个字符 并且padding至 词语最大长度
        超出长度则截取
        :param sent: 传入一句话
        :param word_maxlen: 词语最大长度
        :return: 一句话的字符级分隔
        """
        char_sent = []
        for word in sent:
            # if word in ['[CLS]', '[SEP]', '[PAD]']:
            #     char_sent.append(word)
            #     continue
            char_level = list(word)
            if len(char_level) >= word_maxlen:
                char_level = char_level[:word_maxlen]
            else:
                char_level.extend(['UNK'] * (word_maxlen - len(char_level)))
            char_sent.extend(char_level)
        return char_sent

    def nltk_split(self,sentence):
        # TODO 2）preprocess.py 实现nltk对语料的切分（30'）
        # 载入停用词，stem后切分
        stopword = stopwords.words('english')
        for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
            stopword.append(w)
        splited_sentence = nltk.word_tokenize(sentence)

        porter_stemmer = nltk.stem.PorterStemmer()
        stemed_sentence = [porter_stemmer.stem(word.lower()) for word in splited_sentence]
        return stemed_sentence

    def convert2id(self, sent, maxlen=None, begin=False, end=False):
        """
        分词后的句子打上一些标记，然后处理为词级的句子与字符级的句子
        :param sent:输入的句子
        :param maxlen: 最大句长
        :param begin: 是否需要cls标志
        :param end: 是否需要sep标志
        :return: token化后，一句话词的表现与字符的表现
        """
        word = self.nltk_split(sent)
        word = ['[CLS]'] * begin + word

        if maxlen is not None:
            word = word[:maxlen - 1 * end]
            word += ['[SEP]'] * end
            word += ['[PAD]'] * (maxlen - len(word))
        else:
            word += ['[SEP]'] * end

        # 转换为字符级的句子表示
        char = self.word2char(word, word_maxlen=10)

        word_ids = self.get_word_id(word)
        char_ids = list(map(self.get_char_id, char))

        return word_ids, char_ids

    def get_char_id(self, ch):
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_word_id(self, word_sentence):
        """
        更新词语到id的字典
        :param word_sentence:
        :return:
        """
        cur_word2id_lenth = len(self.word2id)
        for word in word_sentence:
            if word not in self.word2id:
                self.word2id[word] = cur_word2id_lenth
                cur_word2id_lenth += 1

        self.get_id_word()
        tokened_word_sent = [self.word2id[word] for word in word_sentence]
        return tokened_word_sent

    def get_id_word(self):
        # 翻转word2id获得id2word
        self.id2word = dict(zip(self.word2id.values(),self.word2id.keys()))

    def get_dataset(self, ds_fp):
        cs, qs, cc, qc, be = [], [], [], [], []
        for _, c, q, c_char, q_char, b, e in self.get_data(ds_fp):
            cs.append(c)
            qs.append(q)
            cc.append(c_char)
            qc.append(q_char)
            be.append((b, e))
        return map(np.array, (cs, qs, cc, qc, be))

    def get_data(self, ds_fp):
        dataset = pio.load(ds_fp)
        for qid, context, question, text, answer_start in self.iter_cqa(dataset):
            cids_word, cids_char = self.get_sent_ids(context, self.max_clen)
            qids_word, qids_char = self.get_sent_ids(question, self.max_qlen)
            b, e = answer_start, answer_start + len(text)
            if e >= len(cids_word):
                b = e = 0
            yield qid, cids_word, qids_word, cids_char, qids_char, b, e

    def get_sent_ids(self, sent, maxlen):
        return self.convert2id(sent, maxlen=maxlen, end=True)


if __name__ == '__main__':
    p = Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    print(p.encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))
    p.get_embedding_matrix()
