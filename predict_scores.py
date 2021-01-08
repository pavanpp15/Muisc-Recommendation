import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

f_test = open('testItem2.txt', 'r')
pred = open('predictions_1.txt', 'w')

load_model = pickle.load(open('svd_model_final.pkl', 'rb'))

users = list()
tracks = list()
ratings = [0]*120000

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

def get_array(series):
    return np.array([[element] for element in series])


for line in f_test:
    if '|' in line:
        userid = line.strip().split('|')[0]
        continue
    users.append(userid)
    s = line.strip()
    tracks.append(s)
 
data = pd.DataFrame({'user': users, 'itemId': tracks, 'ratings': ratings}) 

x = data.iloc[:, [0,1]].values
y = data.iloc[:, 2].values

mapping_work = get_mapping(data["itemId"])

data["itemId"] = data["itemId"].map(mapping_work)

mapping_users = get_mapping(data["itemId"])

data["itemId"] = data["itemId"].map(mapping_users)

predictions = load_model.predict([get_array(data["itemId"]), get_array(data["user"])])



out = list(predictions)

for i in range(len(users)):
    pred.write(str(users[i])+'|'+str(tracks[i])+'|'+str(out[i][0])+'\n')
