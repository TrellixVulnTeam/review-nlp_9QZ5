import string
import numpy as np
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import model_from_json
import pickle
import h5py
import os


class BaseModel:
    def __init__(self):
        self.lematizer = WordNetLemmatizer()
        self.stop = stopwords.words('english')
        self.transtbl = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        self.topwords = None
        self.word_to_index = None
        self.word_to_vec_map = None

        self.model = None
        self.weights = None
        self.network = None

    # Load Embedding
    def load_embedding(self, topwords_path, word_to_index_path, word_to_vec_map_path, mode='rb'):
        with open(topwords_path, mode) as pkl_file:
            self.topwords = pickle.load(pkl_file)
        with open(word_to_index_path, mode) as pkl_file:
            self.word_to_index = pickle.load(pkl_file)
        with open(word_to_vec_map_path, mode) as pkl_file:
            self.word_to_vec_map = pickle.load(pkl_file)

    # Load Model
    def load_model(self, model_path, weights_path, mode='rb'):

        # Load model structure
        with open(model_path, mode) as fp:
            self.model = model_from_json(fp.read())
        # Load model weights
        self.model.load_weights(weights_path)

    def sentences_to_indices(self, line, maxlen):

        X = np.asarray(line)
        m = X.shape[0]
        X_indices = np.zeros((m, maxlen))
        for i in range(m):
            sentence_words = X[i].lower().split()
            j = 0
            for w in sentence_words:
                if(w in self.topwords and self.word_to_index.get(w)!= None):
                    X_indices[i, j] = self.word_to_index[w]
                    j += 1
                    if j >= maxlen:
                        break
        return X_indices

    # Preprocessing
    def preprocessing(self, line:str) -> str:
        line = line.replace('<br />','').translate(self.transtbl)

        tokens = [self.lematizer.lemmatize(t.lower(), 'v')
                for t in word_tokenize(line)
                if t.lower() not in self.stop]

        return ' '.join(tokens)

    # Predict
    def predict(self, line):
        if not self.model:
            print('Model is not loaded')
            return ""
        line = self.preprocessing(line)
        x = self.sentences_to_indices(np.array([line]), 200)
        sentiment = self.model.predict(x)
        if sentiment > 0.5: return "Positive"
        return "Negative"