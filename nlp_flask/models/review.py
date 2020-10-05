from models.base import BaseModel


class ReviewModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.load_embedding('models/topwords.pkl','models/word_to_index.pkl','models/word_to_vec_map.pkl')
        self.load_model('models/network.json','models/weights.h5')

    def predict(self, line, highlight=True):
        sentiment = super(ReviewModel, self).predict(line)

        #highlight words, hack
        if highlight:
            highlight_words = \
                [w for w in self.preprocessing(line).split()
                if super(ReviewModel, self).predict(w) == sentiment]
            return sentiment, highlight_words
        else:
            return sentiment
