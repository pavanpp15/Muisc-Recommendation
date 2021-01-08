from utils import *
from sklearn.metrics import mean_absolute_error
import pickle


train,test, max_user, max_work, mapping_work = get_data()

#pickle.dump(mapping_work, open('mapping_work.pkl', 'wb'))

pkl_file = open('svd_model_final.pkl', 'wb')

#######################################################################
model = get_model(max_work, max_user)

history = model.fit([get_array(train["itemId"]), get_array(train["userId"])], get_array(train["rating"]), epochs=3,
                    validation_split=0.15, verbose=1)

#model.save_weights("model_1.h5")
pickle.dump(model, pkl_file)

predictions = model.predict([get_array(test["itemId"]), get_array(test["userId"])])

test_performance = mean_absolute_error(test["rating"], predictions)

print(" Test Mae model 1 : %s " % test_performance)
