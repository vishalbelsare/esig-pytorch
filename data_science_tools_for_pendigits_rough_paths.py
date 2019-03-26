# ======================================================================
# import all libraries
# ======================================================================

from free_lie_algebra import *
from logsignature import *

import numpy as np
from esig import tosig
import iisignature
from p_variation import *
from tjl_dense_numpy_tensor import *
import bch

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint as sp_randint

from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC, ClassPredictionError
from PrecisionRecallCurve import PrecisionRecallCurve

from parsing import parser, digit
from plotting import plotter, voronoi
from analysis import training, sampling, testing, classify

from transformers import *
from mpl_toolkits import mplot3d

from cython_sig_distance import sig_distance
import seaborn as sns
import time
from scipy import interp
from itertools import cycle
from urllib.request import urlopen

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

## Simplified pendigits
class Digit:
    def __init__(self, points, digit):
        self.points=points
        self.digit=digit


def get_simplified_pendigits():

     data_train = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra")
     data_test = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes")

     digits=[]
     for line in data_train:
         digitList=line.decode("utf-8").replace(",", " ").replace("\n", "").split()
         number=float(digitList[16])
         points=np.array([[float(digitList[i]), float(digitList[i+1])]
                         for i in range(0,len(digitList)-1, 2)])
         digit=Digit(points, number)
         digits.append(digit)
    
     X_train = [digit.points/100. for digit in digits]
     y_train = [int(digit.digit) for digit in digits]

     digits=[]
     for line in data_test:
         digitList=line.decode("utf-8").replace(",", " ").replace("\n", "").split()
         number=float(digitList[16])
         points=np.array([[float(digitList[i]), float(digitList[i+1])]
                         for i in range(0,len(digitList)-1, 2)])
         digit=Digit(points, number)
         digits.append(digit)
    
     X_test = [digit.points/100. for digit in digits]
     y_test = [int(digit.digit) for digit in digits]

     return X_train, y_train, X_test, y_test


# ======================================================================
# Query data and transforms
# ======================================================================
parse = parser.Parser();
train_digits = parse.parse_file('data/pendigits-train');
test_digits = parse.parse_file('data/pendigits-test')

X_train = [c.curves for c in train_digits]
y_train = [c.label for c in train_digits]
X_test = [c.curves for c in test_digits]
y_test = [c.label for c in test_digits]

# Bounding BOX
def bounding_box_transform(X_list_of_paths):
    X_train_new = []
    for path in X_list_of_paths:
        min_x = np.min([np.min(np.array(stroke).T[0]) for stroke in path])
        min_y = np.min([np.min(np.array(stroke).T[1]) for stroke in path])
        path_new = []
        for stroke in path:
            stroke_x = np.array(stroke).T[0] - min_x
            stroke_y = np.array(stroke).T[1] - min_y
            stroke_new = np.c_[stroke_x, stroke_y]
            stroke_new *= (1./np.max(stroke_new))
            path_new.append(stroke_new.tolist())
        X_train_new.append(path_new)
    return X_train_new

# bound paths
X_train = bounding_box_transform(X_train)
X_test = bounding_box_transform(X_test)

# Stroke transform
stroke_transformer = Stroke_Augment()
X_train_stroke = stroke_transformer.fit_transform(X_train)
X_test_stroke = stroke_transformer.fit_transform(X_test)

# Ink transform
ink_transformer = Ink_Augment()
X_train_ink = ink_transformer.fit_transform(X_train)
X_test_ink = ink_transformer.fit_transform(X_test)

# Pen transform
pen_transformer = Pen_Augment()
X_train_pen = pen_transformer.fit_transform(X_train)
X_test_pen = pen_transformer.fit_transform(X_test)

# ======================================================================
# Data Query: call the relevant data transform
# ======================================================================
def which_transform(flag=None, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    assert flag in [None, 'stroke', 'pen', 'ink']

    if flag is None:
        X_tr = [np.concatenate(x).astype('double') for x in X_train]
        X_te = [np.concatenate(x).astype('double') for x in X_test]

    elif flag == 'stroke':
        X_tr = [x.astype('double') for x in X_train_stroke]
        X_te = [x.astype('double') for x in X_test_stroke]

    elif flag == 'ink':
        X_tr = [x.astype('double') for x in X_train_ink]
        X_te = [x.astype('double') for x in X_test_ink]

    elif flag == 'pen':
        X_tr = [x.astype('double') for x in X_train_pen]
        X_te = [x.astype('double') for x in X_test_pen]

    return X_tr, y_train, X_te, y_test

# ======================================================================
# Tool: Confusion Matrix plotter
# ======================================================================
def plot_confusion_matrix(cm, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def sig_normalize(sig, width=3):
    """ Normalization: 
        multiply each value in level k by k!
    """
    depth = int((np.log(len(sig)*(width-1)+1) / np.log(2)) - 1)
    ind1 = 1
    ind2 = 1
    sig_scaled = np.zeros(len(sig))
    sig_scaled[0] = 1.
    for k in range(1, depth+1):
        ind2 += width**(k)
#         print(ind1, ind2, k)
        sig_scaled[ind1:ind2] = copy.deepcopy(sig[ind1:ind2])*math.factorial(k)
        ind1 = copy.deepcopy(ind2)
    return sig_scaled

def sig_scale(sig, scale_factor, width=3):
    """ Scale:
        multiply each value in level k by scale_factor^k
    """
    depth = int((np.log(len(sig)*(width-1)+1) / np.log(2)) - 1)
    ind1 = 1
    ind2 = 1
    sig_scaled = np.zeros(len(sig))
    sig_scaled[0] = 1.
    for k in range(1, depth+1):
        ind2 += width**(k)
#         print(ind1, ind2, k)
        sig_scaled[ind1:ind2] = copy.deepcopy(sig[ind1:ind2])*(scale_factor**k)
        ind1 = copy.deepcopy(ind2)
    return sig_scaled

# ======================================================================
# Randomized Search 
# ======================================================================
def pendigits_signatures_random_search(depth=5, k_folds=5):    

    names = ["Radius Neighbors",
             "Nearest Neighbors", 
             #"Logistic Regression", 
             "Gaussian Kernel Support Vector Machine",
             "Polynomial Kernel Support Vector Machine",
             "Random Forest", 
             "Neural Net", 
             "Boosting Tree (AdaBoost)"
            ]

    classifiers = [
                   RadiusNeighborsClassifier(weights='distance', algorithm='auto', metric='euclidean', 
                                             outlier_label=np.random.randint(10), metric_params=None, n_jobs=-1),
                   KNeighborsClassifier(algorithm='auto', metric='euclidean', weights='distance', n_jobs=-1),
                   #LogisticRegressionCV(penalty='l2', multi_class='multinomial', cv=k_folds),
                   SVC(kernel='rbf'),
                   SVC(kernel='poly'),
                   RandomForestClassifier(),
                   MLPClassifier(),
                   AdaBoostClassifier()
                  ]

    parameters = [
                  {'radius': np.linspace(start=0.5, stop=2, num=20)}, #radius neighbours
                  {'n_neighbors': [1, 3, 4, 5, 6, 7, 8, 9]}, # k-nn
                  #{}, #logistic
                  {'C': [10, 50, 100, 200], 'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 'shrinking':[True, False]}, # rbf svm
                  {'C': [10, 50, 100, 200], 'gamma':[0.005, 0.01, 0.05, 0.1, 0.5, 1.], 'shrinking':[True, False]}, # poly svm
                  {'n_estimators':[200, 500], 'max_depth':[5, 10, None], 'max_features':[5, 10, 'auto']}, # random forest
                  {'activation':['logistic', 'tanh', 'relu'], 
                   'alpha':[0.001, 0.01, 0.1, 1.], 
                   'learning_rate':['constant', 'adaptive'],
                   'early_stopping':[True, False],
                   'hidden_layer_sizes':[(100,), (100, 100,), (50, 100, 50,)]}, # neural net
                  {'n_estimators':[10, 50, 100, 500], 'learning_rate':[0.01, 0.1, 1.]} # boosting tree
                  ]

    outer_dic = {}
    for transform_flag in [None, 'pen', 'stroke']:
    
        signatures_train = [tosig.stream2sig(p, depth) for p in which_transform(transform_flag)[0]]
        signatures_test = [tosig.stream2sig(p, depth) for p in which_transform(transform_flag)[2]]

        if transform_flag is None:
            print('NO transform')
        else:
            print('{} transform'.format(transform_flag))
    
        if transform_flag is None:
            train = [sig_normalize(x, width=2) for x in signatures_train]
            test = [sig_normalize(x, width=2) for x in signatures_test]
        else:
            train = [sig_normalize(x, width=3) for x in signatures_train]
            test = [sig_normalize(x, width=3) for x in signatures_test]
    
        inner_dic = {}
        for name, clf, param in zip(names, classifiers, parameters):
        
            t = time.time()
        
            if name == 'Logistic Regression':
                clf.fit(train, which_transform(transform_flag)[1])
                score = clf.score(test, which_transform(transform_flag)[3])
            else:
    #             search_model = GridSearchCV(clf, param_grid=param, cv=k_folds, verbose=1, n_jobs=-1)
                search_model = RandomizedSearchCV(clf, param_distributions=param, cv=k_folds, verbose=1, n_jobs=-1)
                search_model.fit(train, which_transform(transform_flag)[1])
                score = search_model.best_estimator_.score(test, which_transform(transform_flag)[3])
            inner_dic[name] = score
            print('{} accuracy: {:.4f} % -- time taken: {:.2f} s'.format(name, score, time.time()-t))
            if name != 'Logistic Regression':
                print('best parameters:')
                print(search_model.best_params_)
            print('\n')
    
        if transform_flag is None:
            outer_dic['No']=inner_dic
        else:
            outer_dic[transform_flag]=inner_dic

# ======================================================================
# Tools to assess how scaling the paths affects classification accuracy
# ======================================================================
def model_call(model):
    if model == 'knn':
        return KNeighborsClassifier(n_neighbors=neighbors, algorithm='auto', metric='euclidean', weights='distance', n_jobs=-1)
    if model == 'rf':
        return RandomForestClassifier(n_estimators=200, max_features='auto')
    if model == 'svm':
        return SVC(C=10, gamma=0.0005, kernel='rbf')

def model_performance(model, model_name, depth):
    
    accuracies = {}
    accuracies_norm = {}
    for transform_flag in [None, 'ink', 'pen', 'stroke']:

        # computing un-normalized signatures
        signatures_train = [tosig.stream2sig(p, depth) for p in which_transform(transform_flag)[0]]
        signatures_test = [tosig.stream2sig(p, depth) for p in which_transform(transform_flag)[2]]

        # computing normalized signatures
        if transform_flag is None:
            w = 2
            t_flag = 'No'
        else:
            w = 3
            t_flag = transform_flag
            
        normalized_signatures_train = [sig_normalize(x, width=w) for x in signatures_train]
        normalized_signatures_test = [sig_normalize(x, width=w) for x in signatures_test]

        neighbors = 5

        sig_accuracy = {}
        norm_sig_accuracy = {}

        it = 0
        for scale_factor in np.linspace(start=1.05, stop=10, num=20):

            if it%5==0:
                print('--- {} model fitting with {}-path transform, degree {} signatures & scale factor: {} ---'.format(model_name, t_flag, depth, scale_factor))

            # define model
            model_sig = model_call(model)
            model_norm_sig = model_call(model)

            # scale signatures
            if transform_flag is None:
                scaled_signatures_train = [sig_scale(sig=sig, scale_factor=scale_factor, width=2) for sig in signatures_train]
                scaled_signatures_test = [sig_scale(sig=sig, scale_factor=scale_factor, width=2) for sig in signatures_test]
                scaled_normalized_signatures_train = [sig_scale(sig=sig, scale_factor=scale_factor, width=2) for sig in normalized_signatures_train]
                scaled_normalized_signatures_test = [sig_scale(sig=sig, scale_factor=scale_factor, width=2) for sig in normalized_signatures_test]
            else:
                scaled_signatures_train = [sig_scale(sig=sig, scale_factor=scale_factor, width=3) for sig in signatures_train]
                scaled_signatures_test = [sig_scale(sig=sig, scale_factor=scale_factor, width=3) for sig in signatures_test]
                scaled_normalized_signatures_train = [sig_scale(sig=sig, scale_factor=scale_factor, width=3) for sig in normalized_signatures_train]
                scaled_normalized_signatures_test = [sig_scale(sig=sig, scale_factor=scale_factor, width=3) for sig in normalized_signatures_test]

            # fit model
            model_sig.fit(scaled_signatures_train, which_transform(transform_flag)[1])
            model_norm_sig.fit(scaled_normalized_signatures_train, which_transform(transform_flag)[1])

            # compute accuracy
            score = model_sig.score(scaled_signatures_test, which_transform(transform_flag)[3])
            score_norm = model_norm_sig.score(scaled_normalized_signatures_test, which_transform(transform_flag)[3])

            # store results
            sig_accuracy[scale_factor] = score
            norm_sig_accuracy[scale_factor] = score_norm

            it+=1

        print('\n')
        accuracies[t_flag] = sig_accuracy
        accuracies_norm[t_flag] = norm_sig_accuracy
        
    return accuracies, accuracies_norm

def plot_results(accuracies, accuracies_norm, model_name, depth):
    flag_names = ['No', 'ink', 'pen', 'stroke']
    k = 0
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for i in [0, 1]:
        for j in [0, 1]:
            name = flag_names[k]
            ax[i,j].plot(accuracies[name].keys(), accuracies[name].values(), label='Un-normalized Signatures')
            ax[i,j].plot(accuracies_norm[name].keys(), accuracies_norm[name].values(), label='Normalized Signatures')
            ax[i,j].legend()
            ax[i,j].set_title('{}-transform'.format(name))
            ax[i,j].set_xlabel('Scale Factor')
            ax[i,j].set_ylabel('Classification Accuracy (%)')
            k+=1
    plt.suptitle('{} classification with un-normalized signatures at level {}'.format(model_name, depth))
    plt.subplots_adjust(hspace = 0.4)
    plt.show()

def random_subsample(size=5, replace=False, transform_flag=None):
    """take random subsample of training data with equal number of instances per class"""
    X = which_transform(transform_flag)[0]
    y = which_transform(transform_flag)[1]
    df_random_sample = pd.DataFrame(data=y, columns=['label'])
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
    df_random_sample = df_random_sample.groupby('label', as_index=False).apply(fn)
    paths_subsample = [X[ind] for ind in df_random_sample.index.levels[1].tolist()]
    labels_subsample = [y[ind] for ind in df_random_sample.index.levels[1].tolist()]
    return paths_subsample, labels_subsample

def compute_centroids(depth=4, transform_flag=None, normalize_flag=True):
    """clustering to find the centroids and use only those as training data"""
    n_classes = 10
    model = KMeans(n_clusters=n_classes, init='k-means++', n_init=n_classes, max_iter=300, tol=0.0001, 
                   precompute_distances='auto', verbose=0, random_state=None, n_jobs=-1, algorithm='auto')
    
    # scale_flag=True
    if transform_flag is None:
        w=2
    else:
        w=3
    paths_train = which_transform(transform_flag)[0]
    signatures_train = [sig_normalize(tosig.stream2sig(p, depth), width=w) for p in paths_train]
    label_train = which_transform(transform_flag)[1]
    clustering = model.fit(signatures_train)
    min_dist = np.min(cdist(signatures_train, model.cluster_centers_, 'euclidean'), axis=1)
    X = pd.DataFrame(label_train, index=range(len(signatures_train)), columns=['True label'])
    Y = pd.DataFrame(min_dist, index=range(len(signatures_train)), columns=['Center'])
    df = pd.concat([X, Y], axis=1)
    grouped = df.groupby(['True label'])
    df_final = grouped.idxmin()
    centroids = [paths_train[n] for n in df_final['Center'].tolist()]
    return centroids

