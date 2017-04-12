from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#nltk.download('all')
nltk.download('punkt')
import sys
''' This is for avoidong error of: 'ascii' codec can't decode byte 0xc2'''
reload(sys)
sys.setdefaultencoding('utf8')


class Preprocessor():
    _lemmatizer = WordNetLemmatizer()
    _stoplist = stopwords.words('english')


    def __init__(self):
        pass

    def init_lists(self,dataPath, label):
        with open(dataPath) as file:
            lines = [line.strip() for line in file]
        # we need at least 1 tab to differentiate content and label
        lines = [line for line in lines if line.count('\t') > 0]

        corpus = [line.split('\t', 1)[1] for line in lines if line.startswith(label)]
        return corpus


    def tokeniz(self,str):
        # tokenizers - spllitting the text by white spaces and punctuation marks
        # lemmatizers - linking the different forms of the same word (for example, price and prices, is and are)
        return [WordNetLemmatizer.lemmatize(self._lemmatizer,word.lower()) for word in word_tokenize(str)
                if 2 < len(word) < 20]
        #return [WordNetLemmatizer.lemmatize("Hello Worls".lower()) for word in word_tokenize(str)]

    def batch_tokeniz(self,corpus):
        return [Preprocessor.tokeniz(self,line) for line in corpus]