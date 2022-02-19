import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#####################################################Data Cleaning##################################################

###############Removing Emojies#####################
col_list = ["type", "tweets"]
data = pd.read_csv("dataset.csv", encoding="utf-8", usecols=col_list)
label = data["type"]
text = data["tweets"]

def remove_emojis(text):
    emoj = re.compile("["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

    return remove_emojis(value)
data["tweets"]=data["tweets"].apply(lambda y:remove_emojis(y))
d=data["tweets"]
#print(d)

#############Removing Punctuation################
def remove_punc(text):
    text_res="".join([c for c in text if c not in string.punctuation])
    return text_res
data["tweets"]=data["tweets"].apply(lambda x:remove_punc(x))
y=data["tweets"]
#print(y)

#################Removing Numbers###################
def remove_numbers(text):
    new = ''.join([i for i in text if not i.isdigit()])
    return new
data["tweets"]=data["tweets"].apply(lambda z:remove_numbers(z))
x=data["tweets"]
#print(x)

############Removing Stopping Words##############
def remove_stoppingwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence

data["tweets"]=data["tweets"].apply(lambda k:remove_stoppingwords(k))
p=data["tweets"]
#print(p)

############ K Fold Cross Validation############
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(text)
le=preprocessing.LabelEncoder()
y=le.fit_transform(label)
kf=RepeatedKFold(n_splits=20,n_repeats=2,random_state=None)
for train_index,test_index in kf.split(text):
    print("Train:",train_index,"Validation:",test_index)
    x_train,x_test=x[train_index],x[test_index]
    y_train,y_test=y[train_index],y[test_index]

print("                                                                         ")
#####################################################Classifiers##################################################

########SVM##########
classfier = svm.SVC(kernel='linear')
classfier.fit(x_train, y_train)
predsvm = classfier.predict(x_test)
print("Accuracy of SVM: ",metrics.accuracy_score(y_test, predsvm))
#Model Precision:Correct positive predictions relative to total positive predictions
precision_svm=metrics.precision_score(y_test, predsvm)
print("Precision:",precision_svm)
#Model Recall: Correct positive predictions relative to total actual positives
Recall_svm=metrics.recall_score(y_test, predsvm)
print("Recall:",Recall_svm)
#F_Measure:-
F1 = 2 * (precision_svm * Recall_svm) / (precision_svm + Recall_svm)
print("F_Measure",F1)
print("                                                                              ")



######Decision tree######
clfDecision= DecisionTreeClassifier(random_state=0)
clfDecision.fit(x_train,y_train)
predtree=clfDecision.predict(x_test)
print("Accuracy of Desicion Tree: ",metrics.accuracy_score(y_test, predtree))
#Model Precision:Correct positive predictions relative to total positive predictions
precision_tree=metrics.precision_score(y_test, predtree)
print("Precision:",precision_tree)
#Model Recall: Correct positive predictions relative to total actual positives
Recall_tree=metrics.recall_score(y_test, predtree)
print("Recall:",Recall_tree)
#F_Measure:-
F1 = 2 * (precision_tree * Recall_tree) / (precision_tree + Recall_tree)
print("F_Measure",F1)
print("                                                                                ")




#####Logistic Regression#####
Our_Model=LogisticRegression(solver='liblinear',C=10,random_state=0)
Our_Model.fit(x_train,y_train)
predict=Our_Model.predict(x_test)
print("Accuracy of Logistic Regression: ",metrics.accuracy_score(y_test, predict))
#Model Precision:Correct positive predictions relative to total positive predictions
precision_Logistic=metrics.precision_score(y_test, predict)
print("Precision:",precision_Logistic)
#Model Recall: Correct positive predictions relative to total actual positives
Recall_logistic=metrics.recall_score(y_test, predict)
print("Recall:",Recall_logistic)
#F_Measure:-
F1 = 2 * (precision_Logistic * Recall_logistic) / (precision_Logistic + Recall_logistic)
print("F_Measure",F1)
print("                                                                                   ")













Saccurecy=classfier.score(x_test,predsvm)
print(Saccurecy)
Taccurecy= clfDecision.score(x_test,predtree)
print(Taccurecy)
Laccurecy=Our_Model.score(x_test,predict)
print(Laccurecy)