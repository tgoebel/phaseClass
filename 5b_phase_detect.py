"""
    Many to one sequence prediction:

    - detect phase arrivals based on systematic moveout (label = 1)
    and
    - differentiate noise (i.e. erroneous picks) from arrival times

    TODO:
    - compare performance of MLP with RNN - LSTM

    -https://machinelearningmastery.com/sequence-prediction-problems-learning-lstm-recurrent-neural-networks/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
#--------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

import ML_CERI8703.ml_utils.gradient_utils as utils
#=================================================
#               files and params
#=================================================
file_train  = 'data/arrival_times_train_ran_1234.txt'
file_test   = 'data/arrival_times_test_ran_1234.txt'

standardize = False
#--------------nn parameters-----------------
n_epochs  = 100
n_hidden  = 40
batch_size= 5
f_dropout = 0.2 # 20 percent dropout after each LSTM layer
#=====================1===========================
#           load training/testing data
#=================================================
mData   = np.loadtxt( file_train)
X_train = mData[:,0:-1]
y_train = mData[:,-1]
print( 'training data shape: ', X_train.shape)
mData   = np.loadtxt( file_test)
X_test = mData[:,0:-1]
y_test = mData[:,-1]

nEv_train  = len( y_train)
nEv_test   = len( y_test)
#---------------standardize----------------------

if standardize == True:
    stdScale = StandardScaler()
    X_train  = stdScale.fit_transform( X_train)
    X_test   = stdScale.transform( X_test)

# convert to 3D arrays
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],  1))
X_test  = np.reshape(X_test,  (X_test.shape[0],   X_test.shape[1],  1))
#=====================3===========================
#           build the RNN
#=================================================
model = Sequential()

model.add( Bidirectional( LSTM( units= n_hidden, return_sequences=True,
                                activation='relu'),
                          input_shape=(X_train.shape[1],1)))
model.add( Dropout( f_dropout))
for s_bool in [True, False]:
    model.add( Bidirectional( LSTM( units = n_hidden, return_sequences = s_bool,
                                    activation='relu')))
    model.add( Dropout(f_dropout))
### add fully connected output layer
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='softmax'))# for multi class labels
# check size of ANN
model.summary()

# train
model.compile( loss='binary_crossentropy',
               optimizer='adam', metrics=['accuracy'])

#=====================4===========================
#           train and test the RNN
#=================================================
dRes = model.fit( X_train, y_train, epochs= n_epochs,
                    validation_split=0.2, batch_size=batch_size,
                    shuffle=True).history
print( '-------------results------------------------')
print( ' keys: ', dRes.keys())
# if standardize == True:
#     X_test[:,:] = stdScale.inverse_transform( X_test[:,:])
y_pred_train = (model.predict(X_train) > 0.5).astype("int32").flatten()
print( 'Train predict: ', y_pred_train[0:30])
print( 'Train Labels: ', y_train[0:30])
scores = model.evaluate(X_train, y_train, verbose=0)
print( "Train Acc.: %.2f%%"%( scores[1]*100))

y_prob = model.predict( X_test, batch_size=batch_size)
y_pred = model.predict_classes(X_test, batch_size=batch_size)
# generate binary output
y_pred2 = (model.predict(X_test) > 0.5).astype("int32")
# for multiclass: y_pred = np.argmax(model.predict(x), axis=-1)
# y_pred = np.ones( nEv_test, dtype = int)
# sel = y_prob < .5
# y_pred[sel] = 0
print( 'Test  probability: ', y_prob[0:5].flatten())
print( 'Test prediction: ', y_pred[0:10].flatten(), y_pred2[0:10].flatten())
print( 'Test true labels', y_test[0:10])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))

#=====================5===========================
#           plot results
#=================================================
plt.figure(1)
# -----1-----cost function-----------------
ax = plt.subplot( 311)
ax.set_title( 'Cost Function')
x_arr = np.arange( 1, n_epochs+1)
ax.semilogy(x_arr, dRes['loss'], '-o', label='Train loss')
ax.semilogy(x_arr, dRes['val_loss'], '--<', label='Validation loss')
ax.legend()
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)


# -----2-----accuracy-----------------
ax2 = plt.subplot( 312)
ax2.plot( x_arr, dRes['accuracy'], '-o', label='Train Acc')
ax2.plot( x_arr, dRes['val_accuracy'], '--<', label='Validation Acc')
ax2.set_xlabel('Epoch', size=15)
ax2.set_ylabel('Accuracy', size=15)
ax2.legend()

# -----3-----confusion matrix-----------------
ax3 = plt.subplot( 313)
# plot confusion matrix
confMat = confusion_matrix(y_test, y_pred)

sns.heatmap( confMat, annot=True, fmt='g', ax=ax3)
ax3.xaxis.set_ticklabels(['Noise', 'Eq.'])
ax3.yaxis.set_ticklabels(['Noise', 'Eq.'])
ax3.set_xlabel('True Label')
ax3.set_ylabel( 'Predict. Label')

plt.savefig("plots/5c_eq_phase_detect.png")
plt.show()







