import csv
import pandas as pd
from collections import OrderedDict
f = open('HW9.txt', 'r')

# scores = list() uncomment
scores  = dict()
binary_scores = dict()
tracks = dict()

d = ['a', 'b', 'c', 'd', 'e', 'f']

data = pd.read_csv('sample_submission.csv')
track_id = list(data['TrackID'])

for h in track_id:
    d = h.strip().split('_')
    ut = d[0]
    sr = d[1]
    if ut not in tracks.keys():
        tracks[ut] = list()
    tracks[ut].append(sr)

i=0
temp_dict = dict()
for line in f:
    s = line.strip().split('|')  # comment this line for 0 and 1
    us = s[0] # comment
    it = s[1] # comment
    score = float(s[2]) # comment
    #score = float(line.strip()[1:-1]) uncomment this line for 0 and 1
    if us not in scores.keys():
        scores[us] = dict()
    scores[us].update({it:score})
    
    
for key, value in scores.items():
        binary_scores[key] = dict()
        track_score_dict = OrderedDict(sorted(value.items(), key=lambda t: t[1]))
        ord_tracks = list(track_score_dict.keys())
        for e in value.keys():
                if len(ord_tracks)!=6:
                    binary_scores[key].update({e:1})
                else:
                    t = ord_tracks.index(e)
                    if t<3:
                        binary_scores[key].update({e:0})
                    else:
                        binary_scores[key].update({e:1})
    
    
with open('HW9.csv', 'w', newline='') as ff:
    writer = csv.writer(ff)
    writer.writerow(["TrackID", "Predictor"])
    for i in tracks.keys():
        for n in tracks[i]:
            try:
                t_score = binary_scores[i][n]
                writer.writerow([str(i)+'_'+str(n), t_score])
            except:
                writer.writerow([str(i)+'_'+str(n), str(0)])

    