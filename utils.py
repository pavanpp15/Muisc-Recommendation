import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import defaultdict
from sklearn.model_selection import train_test_split

def get_mapping(series):
    occurances = defaultdict(int)
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i

    return mapping


def get_data():
    data = pd.read_csv("track_ratings_1.csv")
    
    data['rating'] = data['rating']/20

    mapping_work = get_mapping(data["itemId"])

    data["itemId"] = data["itemId"].map(mapping_work)

    mapping_users = get_mapping(data["itemId"])

    data["itemId"] = data["itemId"].map(mapping_users)
    
    x, y = train_test_split(data, test_size = 0.15)

    max_user = max(data["userId"].tolist() )
    max_work = max(data["itemId"].tolist() )


    return x, y, max_user, max_work, mapping_work



def get_model(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = Dense(10, activation="relu")(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])

