from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext

sc = SparkContext()

train_data = sc.textFile("trainData.data")
train_ratings = train_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

rank = 30
numIterations = 25
model = ALS.train(train_ratings, rank, numIterations, nonnegative=True)

testFile = sc.textFile("testData.data")

test_ratings = testFile.map(lambda l: l.split(',')) .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))



testdata = test_ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

f = open('spark_predictions.txt', 'w')

op = sorted(ratesAndPreds.collect(), key = lambda r: int(r[0][0]))

for x in op:
    if x[1][1]!='NaN':
        f.write(str(x[0][0])+'|'+str(x[0][1])+'|'+str(int(x[1][1]))+'\n')
    else:
        f.write(str(x[0][0])+'|'+str(x[0][1])+'|'+str(0)+'\n')
"""
prediction = sorted(predictions.collect(), key = lambda r: int(r[0]))

with open("spark_prediction.txt","w") as predFile:
    for line in prediction:
        
        if line[2]!=line[2]:
            temp_str = "0"
        else:
            temp_str = str(int(line[2]))
        predFile.write(str(line[0])+"|"+str(line[1])+"|"+temp_str+"\n")
        #predFile.write("["+str(temp_str)+']\n')

"""
