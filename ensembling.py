import numpy as np

P1='prediction_1.txt'
P2='prediction_2.txt'
P3='prediction_3.txt'
P4='prediction_4.txt'
P5='prediction_5.txt'
P6='prediction_6.txt'

predicitons = [P1, P2, P3, P4, P5, P6]

fRating=open('ratings.txt','r')
fOut=open('ensemble_score.txt','w')

YX=[0]*6
YX[0]=(2*0.62138-1)*120000
YX[1]=(2*0.69603-1)*120000
YX[2]=(2*0.69624-1)*120000
YX[3]=(2*0.86746-1)*120000
YX[4]=(2*0.87193-1)*120000
YX[5]=(2*0.87374-1)*120000


y1 = list()
y2 = list()
y3 = list()
y4 = list()
y5 = list()
y6 = list()

y_matrix = [y1, y2, y3, y4, y5, y6]

for i in range(len(predictions)):
    with open(predictions[i]) as f1:
        Y1 = f1.read().splitlines()
    for item in Y1:
        y_matrix[i].append(float(item))
    f1.close()

for i in range(len(y_matrix)):
    y_matrix[i]=np.matrix(y_matrix[i])

Y1=2*y1.T-1
Y2=2*y2.T-1
Y3=2*y3.T-1
Y4=2*y4.T-1
Y5=2*y5.T-1
Y6=2*y6.T-1

Y=np.concatenate((Y1, Y2, Y3, Y4, Y5, Y6), axis=1)
YY=np.dot(Y.T,Y)
inv_YY= np.linalg.inv(YY)
YX= np.matrix(YX)
YX=YX.T
A=inv_YY*YX
NEW=Y1*A[0]+Y2*A[1]+Y3*A[2]+Y4*A[3]+Y5*A[4]+Y6*A[5]

i=0
for line in fRating:
        arr_test=line.strip().split('|')
        userID = arr_test[0]
        trackID= arr_test[1]
        outStr=str(userID) + '|' + str(trackID)+ '|' + str(NEW.item(i))
        fOut.write(outStr + '\n')
        i=i+1

import heapq as hp
import numpy as np

fRating=open('output.txt','r')
fOut=open('prediction_ensemble.txt','w')

score_vec=[0]*6
lastUserID=-1
for line in fRating:
        arr_test=line.strip().split('|')
        userID = int(arr_test[0])
        trackID= arr_test[1]
        score= float(arr_test[2])
        if userID != lastUserID:
                ii=0
        weight=[0.97,0.96, 0.99, 0.998, 0.999, 1]
        score_vec[ii]= score
        ii=ii+1
        lastUserID=userID
        if ii==6:
          rec=[0]*6   
          n_largest=hp.nlargest(3,zip(score_vec, itertools.count()))
          [first,index1]=n_largest[0]
          [second,index2]=n_largest[1]
          [third,index3]=n_largest[2]
          rec[index1]=1
          rec[index2]=1
          rec[index3]=1

          for nn in range(0,6):
            #outStr=str(userID) + '|' + str(trackID)+ '|' + str(rec[nn])
            outStr= str(rec[nn])
            fOut.write(outStr + '\n')

fRating.close()
fOut.close()
