from keras.models import model_from_json
import numpy
import os

# model and weights location
root = '/srv/home/nkrakauer/chemnetEuler/'

# load json and create model
json_file = open(root+'freesolv_expt_architecture_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(root+"freesolv_expt_bestweights_trial_1_0.hdf5 ")
print("Loaded model from disk")


