import numpy as np
from gensim.models import Word2Vec

# TODO: Make an abstract class and use inheritance
# TODO: Add Comment
# TODO: Add Word2Vec Param to Class


class Word2VecVectorizer:
    def __init__(self, model_name, data=None):
        if data is None:
            self.model = Word2Vec.load('Word2Vec' + model_name)
        else:
            self.model = Word2Vec(data, workers=4, iter=100)
            self.model.save('Word2Vec'+model_name)

    def represent(self, data):
        ret = np.zeros((len(data), self.model.vector_size), dtype=float)
        for idx, row in enumerate(data):
            tmp = np.zeros(self.model.vector_size, dtype=float)
            cnt = 0
            for word in row:
                if word in self.model.wv.vocab:
                    cnt += 1
                    tmp += self.model[word]
            if cnt != 0:
                tmp /= cnt
            ret[idx] = tmp
        return ret


class BoWVectorizer:
    def __init__(self, init_data, tfidf=True, min_freq=5, max_vocab=10000, normalize=None):
        self.words_idx = {}
        self.words_df = {}
        self.words_count = {}
        self.words_number = 0
        self.tfidf = tfidf
        self.normalize = normalize
        for row in init_data:
            unique = set()
            for word in row:
                if word not in self.words_df:
                    self.words_df[word] = 0
                    self.words_count[word] = 0

                if word not in unique:
                    self.words_df[word] += 1
                    unique.add(word)

                self.words_count[word] += 1

        for word in self.words_df:
            self.words_df[word] = np.log2(len(init_data) / self.words_df[word])

        self.words_count = sorted(self.words_count.items(), key=lambda x: x[1], reverse=True)
        if len(self.words_count) > max_vocab:
            self.words_count = self.words_count[:max_vocab]
        while self.words_count[-1][1] < min_freq:
            self.words_count.pop()

        self.words_number = len(self.words_count)
        self.vocab = []
        for idx, word in enumerate(self.words_count):
            self.vocab.append(word[0])
            self.words_idx[word[0]] = idx

    def represent(self, data):
        ret = np.zeros((len(data), self.words_number))
        for idx, row in enumerate(data):
            for word in row:
                if word in self.words_idx:
                    if self.tfidf:
                        ret[idx, self.words_idx[word]] += 1
                    else:
                        ret[idx, self.words_idx[word]] = 1
            if self.tfidf:
                unique = set()
                for word in row:
                    if word not in self.words_idx:
                        continue
                    if word not in unique:
                        unique.add(word)
                        ret[idx, self.words_idx[word]] = np.log10(ret[idx, self.words_idx[word]])*self.words_df[word]

        if self.normalize == 'l2':
            b = np.linalg.norm(ret, axis=1, ord=2)
            for i in range(len(ret)):
                if b[i] != 0:
                    ret[i] = ret[i] / b[i]

        return ret

