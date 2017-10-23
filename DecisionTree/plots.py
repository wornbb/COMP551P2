import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from Utility.csv_reader import read_csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle



# this code was used only to generate OCR and confusion matrix plot. Don't bother reading it.



# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, y_pred)
validation_y = [2419,
4636,
4906,
5065,
5213,
5295,
 5353,
 5424,
 5472,
 5452,
 5446,
 5398,
 5379,
 5380,
 5380,
 5380,
 5380,
 5380]
validation_x = [1,5,6,7,8,9,10,11,12,13,14,21,31,41,51,61,71,81]
f = plt.figure(1)
plt.plot(validation_x,[acc/6844 for acc in validation_y],label='OCR')
plt.title('OCR')
g = plt.figure(2)
decision_tree = pickle.load( open( "depth_12", "rb" ) )

[index,x,y] = read_csv('cleaner1.csv',3)
x = x[1:len(x)]
y = y[1:len(y)]
unique, counts = np.unique(y,return_counts = True)
max_sample_per_class = counts.min()
table = np.column_stack((x,y))
balanced_table = np.array([])
remain_table = np.array([])
for label in unique:
    index = (y==label).nonzero()[0]
    balanced_x = x.take(index,axis=0)[0:max_sample_per_class]
    remain_x = x.take(index,axis=0)[max_sample_per_class:len(x)]
    balanced_y = y.take(index,axis=0)[0:max_sample_per_class]
    remain_y = y.take(index, axis=0)[max_sample_per_class:len(y)]
    add_table = np.column_stack((balanced_x,balanced_y))
    add_remain_table = np.column_stack((remain_x,remain_y))
    if balanced_table.size:
        balanced_table = np.vstack((balanced_table,add_table))
    else:
        balanced_table = add_table
    if remain_table.size:
        remain_table = np.vstack((remain_table,add_remain_table))
    else:
        remain_table = add_remain_table

x = remain_table[:,0]
y = remain_table[:,1]
sp = int(len(x)/10)
l = len(x)
temp = np.column_stack((x,y))
np.random.shuffle(temp)
x = temp[sp:len(x),0]
y = temp[sp:len(y),1]
x_pred = temp[1:sp,0]
y_test = temp[1:sp,1]
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
counts = ngram_vectorizer.fit_transform(x)
feature_names = ngram_vectorizer.get_feature_names()

fe = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1),vocabulary=feature_names)
counts_pred = fe.fit_transform(x_pred)
y_pred = decision_tree.predict(counts_pred)
cnf_matrix = confusion_matrix(y_test.astype(int),y_pred)
np.set_printoptions(precision=2)
class_names = ['Slovak','French','Spanish','German','Polish']
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
