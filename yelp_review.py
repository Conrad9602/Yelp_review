import matplotlib as matplotlib
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report

nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# _____________________import data
yelp_df = pd.read_csv("yelp.csv")

# _____________________visualiza data
# get the length of each messages
yelp_df['length'] = yelp_df['text'].apply(len)
#plot distribution of length of reviews
yelp_df['length'].plot(bins=100, kind='hist')
plt.show()
plt.title('distribution of length of reviews')

# check info of these messages
print('check info of these messages')
print(yelp_df.length.describe())
print()

# plot the stars of each reviews
sns.countplot(y = 'stars', data=yelp_df)
print('check the stars of each reviews')
plt.show()

g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5)
g.map(plt.hist, 'length', bins = 20, color = 'r')
plt.show()

#compare the one and five star reviews
yelp_df_1 = yelp_df[yelp_df['stars']==1]
yelp_df_5 = yelp_df[yelp_df['stars']==5]
yelp_df_1_5 = pd.concat([yelp_df_1 , yelp_df_5])
print('1 star ratio')
print( '1-Stars percentage =', (len(yelp_df_1) / len(yelp_df_1_5) )*100,"%")
print('5 star ratio')
print( '5-Stars percentage =', (len(yelp_df_5) / len(yelp_df_1_5) )*100,"%")
sns.countplot(yelp_df_1_5['stars'], label = "Count")
plt.show()


# ____________EXERCISES to remove punctuations
#Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'
#Test_punc_removed = [char for char in Test if char not in string.punctuation]
#Test_punc_removed_join = ''.join(Test_punc_removed)
#Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

# ____________create testing and training dataset
# ————————————— apply NLP to data

# define a pipeline to clean up all the messages
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

#test the newly added function
yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
# test showing the cleaned up version
print(yelp_df_clean[0]) # show the cleaned up version
# test showing the original version
print(yelp_df_1_5['text'][0]) # show the original version

# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])

# ________________ training the model with all dataset
NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values
NB_classifier.fit(yelp_countvectorizer, label)

# ________________ divide data into training and testing each prior to training
X = yelp_countvectorizer
y = label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# evaluate the model
# predicting the train set results
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
plt.figure()
sns.heatmap(cm, annot=True)
plt.title('classification of training results')
plt.show()

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
plt.figure()
sns.heatmap(cm, annot=True)
plt.title('classification of testing results')
plt.show()

# print the result
print(classification_report(y_test, y_predict_test))