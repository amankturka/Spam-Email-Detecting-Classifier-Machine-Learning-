import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Read, shuffle and pre-process data
print ("Read spambase data ")
dataset = genfromtxt('spambase.data', delimiter=',') 
lengthData = len(dataset)
tar = dataset[:, -1]

np.random.shuffle(dataset)
attributes = preprocessing.scale(dataset[:, 0:-1])
tar = dataset[:, -1]
#split data into training and testing dataset
train_X, test_X, train_Y, test_Y = train_test_split(
    attributes, tar, test_size=0.5, random_state=17)

TotalSpam = 0
train_length = len(train_X)
for eachrow in range(train_length):
    if train_Y[eachrow] == 1:
        TotalSpam += 1		
spam_probability = float(TotalSpam) / train_length #total probabilities for the spam 
nonspam_probability = 1 - spam_probability #total probabilities for the non-spam 
print("Spam probability : \t",spam_probability)
print("Non-spam probability : \t",nonspam_probability)

spam_mean,spam_standev,nonspam_mean,nonspam_standev  = [], [],[],[]
for attributes in range(0,57):
    spam_val,nonspam_val = [],[]
    for eachrow in range(0, train_length):
        if (train_Y[eachrow] == 1):
            spam_val.append(train_X[eachrow][attributes])
        else :
           nonspam_val.append(train_X[eachrow][attributes])
    spam_mean.append(np.mean(spam_val))
    nonspam_mean.append(np.mean(nonspam_val))
    spam_standev.append(np.std(spam_val))
    nonspam_standev.append(np.std(nonspam_val))

for feature in range(0,57):
    if(spam_standev[feature]==0):
        spam_standev[feature] = .0001
    if(nonspam_standev[feature]==0):
        nonspam_standev[feature]=.0001
		
def cal_accuracy_precision_recall(tar, predicted_values, threshold_value):
    pos_true,pos_false,neg_true,neg_false = 0,0,0,0
    for eachrow in range(len(predicted_values)):
        if (predicted_values[eachrow] > threshold_value and tar[eachrow] == 1)  :
            pos_true += 1
        elif (predicted_values[eachrow] > threshold_value and tar[eachrow] == 0 )  :
            pos_false += 1
        elif (predicted_values[eachrow] <= threshold_value and tar[eachrow] == 1 )  :
            neg_false += 1
        elif (predicted_values[eachrow] <= threshold_value and tar[eachrow] == 0 )  :
            neg_true += 1
    accuracy = float(pos_true + neg_true) / len(predicted_values)
    recall = float(pos_true) / (pos_true + neg_false)
    precision = float(pos_true) / (pos_true + pos_false)
    return  accuracy, recall, precision
	
#probability calculation and predicting classes 
probability_spam,probability_non_spam = 0,0
pred = []
for eachrows in range(0,len(test_X)):
    spamProb,nonspamProb = [],[]
    cal1,cal2,cal3,cal4 = 0,0,0,0
    for attributes in range(0,57):
        cal1 = float(1)/ (np.sqrt(2 * np.pi) * spam_standev[attributes])
        cal2 = (np.e) ** - (((test_X[eachrows][attributes] - spam_mean[attributes]) ** 2) / (2 * spam_standev[attributes] ** 2))
        spamProb.append(cal1 * cal2)
        cal3 = float(1)/ (np.sqrt(2 * np.pi) * nonspam_standev[attributes])
        cal4 = (np.e) ** - (((test_X[eachrows][attributes] - nonspam_mean[attributes]) ** 2) / (2 * nonspam_standev[attributes] ** 2))
        nonspamProb.append(cal3 * cal4)
    probability_spam = np.log(spam_probability) + np.sum(np.log(np.asarray(spamProb)))
    probability_non_spam = np.log(nonspam_probability) + np.sum(np.log(np.asarray(nonspamProb)))
    output = np.argmax([probability_non_spam, probability_spam])
    pred.append(output)
acc,rec,pre = cal_accuracy_precision_recall(test_Y, pred, 0)
print("Confusion Matrix:\n",metrics.confusion_matrix(test_Y, pred))
print ("Precision : \t", pre)
print ("Accuracy : \t",acc)
print ("Recall : \t",rec)