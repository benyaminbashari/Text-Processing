import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import time
import re


class TextPreprocessor:
    """
    This class can be used for cleaning texts and also extracting hashtags and mentions
    Usage: 
    Make an object of this class with the desired parameters
    Call preprocess method on the object
    """
    def __init__(self, stem=True, rm_punctuations=True, rm_stopwords=True, rm_digits=True, rm_hashtags=True, rm_mentions=True, rm_links=True):
        """
        Following parameters determines the behaviour of pre-processing steps
        rm stands for remove, default parameter for all is True
        :param stem:
        :param rm_punctuations:
        :param rm_stopwords:
        :param rm_digits:
        :param rm_hashtags:
        :param rm_mentions:
        :param rm_links:
        """
        self.stem = stem
        self.rm_punctuations = rm_punctuations
        self.rm_stopwords = rm_stopwords
        self.rm_digits = rm_digits
        self.rm_hashtags = rm_hashtags
        self.rm_mentions = rm_mentions
        self.rm_links = rm_links
        self._stop = set(stopwords.words('english'))
        self._punctuation = set(string.punctuation)
        self._punctuation.remove('#')
        self._punctuation.remove('@')
        self._sno_stem = SnowballStemmer('english')

    def preprocess(self, texts, workers=1, show=False):
        """
        :param texts: List of all texts
        :param workers: number of threads to run this method
        :param show: Not Complete (TODO)
        :return: list of lists each inner list is word separated sentence
        """
        beg_time = time.time()
        if show:
            print("Total Number of Text to Preprocess:", len(texts))
        if workers == 1:
            ans = []
            for text in texts:
                ans.append(self._preprocess_text(text))
        else:
            with Pool(processes=workers) as pool:
                ans = pool.map(self._preprocess_text, texts)

        if show:
            print("Preprocess Completed in %f seconds" % (time.time()-beg_time))

        return ans

    def _preprocess_text(self, text):
        text = text.lower()
        if self.rm_links:
            text = self._remove_links(text)
        text = self._remove_non_ascii(text)
        if self.rm_punctuations:
            text = self._remove_punctuations(text)
        if self.rm_digits:
            text = self._remove_digits(text)

        lst = word_tokenize(text, 'english')

        if self.rm_hashtags:
            lst = self._remove_hashtags(lst)
        else:
            new_lst = []
            for elem in lst:
                if elem != '#':
                    new_lst.append(elem)
            lst = new_lst
        if self.rm_mentions:
            lst = self._remove_mentions(lst)
        else:
            new_lst = []
            for elem in lst:
                if elem != '@':
                    new_lst.append(elem)
            lst = new_lst
        if self.rm_stopwords:
            lst = self._remove_stopwords(lst)
        if self.stem:
            lst = self._stemmer(lst)
        return lst

    @staticmethod
    def extract_hashtags(texts, unique=False):
        """
        :param texts: in form of list of string
        :param unique: if True for each text the hashtags will be unique
        :return: list of lists each inner list contains hashtags of that string
        """
        ret = []
        for text in texts:
            lst = word_tokenize(text, 'english')
            temp = []
            temp_set = set()
            for i in range(len(lst)):
                if i - 1 >= 0 and lst[i - 1] == '#':
                    str = '#'+lst[i]
                    temp.append(str)
                    temp_set.add(str.lower())
            if unique:
                ret.append(list(temp_set))
            else:
                ret.append(temp)
        return ret

    @staticmethod
    def extract_mentions(texts, unique=False):
        """
        :param texts: in form of list of string
        :param unique: if True for each text the mentions will be unique
        :return: list of lists each inner list contains mentions of that string
        """
        ret = []
        for text in texts:
            lst = word_tokenize(text, 'english')
            temp = []
            temp_set = set()
            for i in range(len(lst)):
                if i - 1 >= 0 and lst[i - 1] == '@':
                    str = '@'+lst[i]
                    temp.append(str)
                    temp_set.add(str.lower())
            if unique:
                ret.append(list(temp_set))
            else:
                ret.append(temp)
        return ret

    def _remove_punctuations(self, text):
        ret = ""
        for ch in text:
            if ch not in self._punctuation:
                ret += ch
            else:
                ret += ' '
        return ret

    def _remove_non_ascii(self, text):
        ret = ''
        for ch in text:
            if 0 <= ord(ch) <= 127:
                ret += ch
        return ret

    def _remove_links(self, text):
        text = re.sub(r'http\S+', '', text)
        return text

    def _remove_digits(self, text):
        ret = ''
        for ch in text:
            if not ord('0') <= ord(ch) <= ord('9'):
                ret += ch
        return ret

    def _remove_stopwords(self, lst):
        return list(filter(lambda x: x not in self._stop, lst))

    def _remove_hashtags(self, lst):
        ret = []
        for i in range(len(lst)):
            if lst[i] != '#' and (i-1 < 0 or lst[i-1] != '#'):
                ret.append(lst[i])
        return ret

    def _remove_mentions(self, lst):
        ret = []
        for i in range(len(lst)):
            if lst[i] != '@' and (i - 1 < 0 or lst[i - 1] != '@'):
                ret.append(lst[i])
        return ret

    def _stemmer(self, text_as_list):
        ret = []
        for word in text_as_list:
            ret.append(self._sno_stem.stem(word))
        return ret



