import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import plot_roc_curve
from sklearn import datasets, metrics, model_selection

from keras.models import Sequential
from keras.layers import Dense,  Activation
from keras.regularizers import L1L2
from keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns

####Reading data from csv files of neighborhoods
output_feats = 'flat_neighbourhoods_1.csv'
df_iter = pd.read_csv(output_feats,header=None,index_col=0)
output_feats2 = 'flat_neighbourhoods_3.csv'
df_iter1 = pd.read_csv(output_feats,header=None,index_col=0)
df_iter = df_iter.append(df_iter1)
len(df_iter)

#print(df_iter)
X_all = df_iter.iloc[:, :63]
y_all = df_iter.iloc[:, 63:]
print(len(X_all))
print(len(y_all))

X_all = StandardScaler().fit_transform(X_all)
X_sel,X_test, y_sel, y_test = \
        train_test_split(X_all, y_all, test_size=40,random_state=42)

print(len(X_sel),len(X_test))

####Definition of MLP-LR classifier in Keras
def MLP_LR_NN (X_train,y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    model = Sequential()

    model.add(Dense(63, activation='relu',kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                   input_dim = len(X_train[0])))
    model.add(Dense(1,  # output dim is 2, one score per each class
                    activation='sigmoid',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                    input_dim=20))  # input dimension = number of features your data has
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(X_train, encoded_Y, epochs=70, batch_size = 20,verbose = 0)
    return model,history

####Plotting accuracy
def plot_data(dat,plotWhat,y_label,title):
    
    d1 = pd.DataFrame(columns = ['classifier', 'n', plotWhat, 'color'])

    k = 0
    for ni in np.unique(dat['n']):
        for cl in np.unique(dat['classifier']):

            tmp = dat[np.logical_and(dat['classifier'] == cl,dat['n'] == ni)][['n', plotWhat]]
            se = stats.sem(tmp[plotWhat].astype(float))
            list(tmp.mean())
            d1.loc[k] = [cl] + list(tmp.mean()) + [names[cl]]
            k += 1


    sns.set(style="darkgrid", rc={'figure.figsize':[12,8], 'figure.dpi': 300})
    fig, ax = plt.subplots(figsize = (8,6))

    for key in names.keys():
        grp = d1[d1['classifier'] == key]
        ax = grp.plot(ax=ax, kind='line', x='n', y=plotWhat, label=key, \
                c = names[key], alpha =0.65)
        ax.set_xscale('log')

    plt.legend(loc='top left',title='Algorithm')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Number of Training Samples')
    plt.show()
    
    ###Training
names = {"MLP-LR": "black", "LR":"blue", "MLP-relu-LR":'red'}

ncores=1
num_runs=2
n_est=100
filename = 'Non_linear_classification2.csv'

classifiers = [MLPClassifier(hidden_layer_sizes = 4,activation = 'logistic', alpha=1, max_iter=1000),
               LogisticRegression(max_iter=2000),
               MLP_LR_NN (X_train,y_train)
    ]

# Train each classifier on each data set size, then test
## Prep output file:
f = open(filename, 'w+')
f.write("classifier,n,Accuracy,trainTime,testTime,iterate\n")
f.flush()

ns = np.array([10,100,500,1000,1500,2000,2500,2800])
runList = [(clf) for clf in zip(classifiers, [key for key in names])]
for n in tqdm(ns):
    print (n)
    for iteration in tqdm(range(num_runs)):
        
        #sampling of data
        X_train = X_sel[:n]
        y_train = np.array(y_sel[:n]).ravel()
        
        #X = X_train2[:n,:]
        #y = Y_train2[:n]

        print (iteration)
        for clf in tqdm(runList):
            
            if(clf[1] == "MLP-relu-LR" ):
                trainStartTime = time.time()
                cls,his = MLP_LR_NN (X_train,y_train)
                trainEndTime = time.time()
                trainTime = trainEndTime - trainStartTime
                encoder = LabelEncoder()
                encoder.fit(y_test)
                en_y_test = encoder.transform(y_test)
                
                testStartTime = time.time()
                score = cls.evaluate(X_test, en_y_test, batch_size=20)
                testEndTime = time.time()
                testTime = testEndTime - testStartTime
                acc=score[1]
            else:
                #training
                trainStartTime = time.time()
                clf[0].fit(X_train, y_train)
                trainEndTime = time.time()
                trainTime = trainEndTime - trainStartTime
                #prediction
                testStartTime = time.time()
                out = clf[0].predict(X_test)
                testEndTime = time.time()
                testTime = testEndTime - testStartTime
                #accuracy
                acc = accuracy_score(y_test,out)

            
            #writing to file
            ####("variable,num of training samples,Lhat,avg precision,trainTime,testTime,iterate")
            f.write(f"{clf[1]}, {n}, {acc:2.9f}, {trainTime:2.9f}, {testTime:2.9f}, {iteration}\n")
            f.flush()
f.close()

#names = {"kNN": "black", "RF":"blue", "MLP":"green", "S-RerF":"red"}
#filename = 'Non_linear_classification1.csv'
dat = pd.read_csv(filename)

##Plot Average Precision
plot_data(dat,'Accuracy','Accuracy','MLP-LR vs LR classification')
