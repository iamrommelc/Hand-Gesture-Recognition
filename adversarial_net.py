

!pip install -qq -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
#!pip install git+https://github.com/cleverhans-lab/cleverhans.git#egg=cleverhans
import sys
sys.path.append('/content/src/cleverhans')
import cleverhans

from google.colab import drive 
drive.mount('/mntDrive')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils import np_utils
from cleverhans.tf2.attacks import fast_gradient_method, \
    basic_iterative_method, momentum_iterative_method
np.random.seed(10)



COL_NAME = [ 'a','b','c','d','e','a2','b2','c2','d2','e2','afd','bfd','cfd','dfd','efd','asd','bsd','csd','dsd','esd','Class_Label']


def get_ds():
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import SMOTE 
    #dataset import
    dataset = pd.read_csv('enter dataset path')
    dataset.head(10)

    X_sm = dataset.iloc[:,:20].values
    y_sm = dataset.iloc[:,20:21].values
    

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_sm = sc.fit_transform(X_sm)

    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    y_sm = ohe.fit_transform(y_sm).toarray()

    sm = SMOTE(random_state=2)
    X_sm, y_sm = sm.fit_resample(X_sm, y_sm)

    return X_sm, y_sm

def create_tf_model():
    
    return model

def gen_tf2_fgsm_attack(org_model, x_test):
    """ This method creates adversarial examples with fgsm """
    logits_model = tf.keras.Model(org_model.input, model.layers[-1].output)

    epsilon = 0.1
    adv_fgsm_x = fast_gradient_method.fast_gradient_method(logits_model,
                                      x_test,
                                      epsilon,
                                      np.inf,
                                      targeted=False)
    return adv_fgsm_x

def gen_tf2_bim(org_model, x_test):
    """ This method creates adversarial examples with bim """
    logits_model = tf.keras.Model(org_model.input, model.layers[-1].output)

    epsilon = 0.1
    adv_bim_x = basic_iterative_method.basic_iterative_method(logits_model,
                                       x_test,
                                       epsilon,
                                       0.1,
                                       nb_iter=10,
                                       norm=np.inf,
                                       targeted=False)
    return adv_bim_x

def gen_tf2_mim(org_model, x_test):
    """ This method creates adversarial examples with mim """
    logits_model = tf.keras.Model(org_model.input, model.layers[-1].output)

    epsilon = 0.1
    adv_mim_x = momentum_iterative_method.momentum_iterative_method(logits_model,
                                          x_test,
                                          epsilon,
                                          0.1,
                                          nb_iter=100,
                                          norm=np.inf,
                                          targeted=True)
    return adv_mim_x

EPOCH = 100
TEST_RATE = 0.2
VALIDATION_RATE = 0.2

X, y = get_ds()

num_class = len(np.unique(y))

attack_functions = [gen_tf2_fgsm_attack, \
                    gen_tf2_mim, \
                    gen_tf2_bim ]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

import keras
from keras.models import Sequential
from keras.layers import Dense
    # Neural network
model = Sequential()
model.add(Dense(64, input_dim=20, activation="relu"))
model.add(Dense(32, input_dim=16, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=100, batch_size=64)




y_pred = model.predict(X_test)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))




from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2', 'class 3','class 4', 'class 5', 'class 6', 'class 7','class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14','class 15', 'class 16', 'class 17', 'class 18','class 19', 'class 20', 'class 21', 'class 22', 'class 23']
print(classification_report(test, pred, target_names=target_names))
print([target_names])
from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)


history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)



import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

from sklearn.metrics import precision_recall_fscore_support
for attack_function in attack_functions:
        print("*"*20)
        print("Attack function is ", attack_function)
        
        history = model.fit(X_train, y_train, epochs=EPOCH,
                            batch_size=16, verbose=0,
                            validation_split=VALIDATION_RATE)
        
        X_adv_list = []
        y_adv_list = []

        adv_x = attack_function(model, X_test)
        
        y_pred = model.predict_classes(adv_x)
        cm_adv = confusion_matrix(test, y_pred)
        
        print("*"*20)
        print("Attacked confusion matrix")
        
       # print(cm_adv)
        

        print("Adversarial training")
        # define the checkpoint

        adv_x = attack_function(model, X_train)
        adv_x_test = attack_function(model, X_test)

        concat_adv_x = np.concatenate([X_train, adv_x])
        concat_y_train = np.concatenate([y_train, y_train])

        history = model.fit(concat_adv_x, concat_y_train, epochs=EPOCH,
                            batch_size=16, verbose=0,
                            validation_data=(adv_x_test, y_test))

       
        y_pred = model.predict_classes(adv_x_test)
        cm_adv = confusion_matrix(test, y_pred)
        
        print("*"*20)
        print("Attacked confusion matrix - adv training")
        print(cm_adv)
        print(accuracy_score(test, y_pred))
        print(precision_recall_fscore_support(test, y_pred, average='macro'))
        from sklearn.metrics import classification_report
        target_names = ['class 0', 'class 1', 'class 2', 'class 3','class 4', 'class 5', 'class 6', 'class 7','class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14','class 15', 'class 16', 'class 17', 'class 18','class 19', 'class 20', 'class 21', 'class 22', 'class 23']
        print(classification_report(test, y_pred, target_names=target_names))
        print([target_names])
        from sklearn.metrics import accuracy_score
        a = accuracy_score(y_pred,test)
        print('Accuracy is:', a*100)