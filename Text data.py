#!/usr/bin/env python
# coding: utf-8

# # Alzheimer's diesease prediction using text data set 

# In[28]:


import pandas as pd
import nltk
import re
import contractions


# In[27]:


pip install contractions


# In[17]:


column_names=['label','text']


# In[19]:


data=pd.read_csv(r"C:\Users\HP\Downloads\data (2).csv",encoding='unicode_escape',header=None,names=column_names)


# In[20]:


data


# # Preprocessing

# In[23]:


# Function to remove punctuations
def remove_punctuations(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text


# In[24]:


# Function to remove words and digits containing digits
def remove_words_with_digits(text):
    cleaned_text = re.sub(r'\w*\d\w*', '', text)
    return cleaned_text


# In[25]:


# Function to remove extra spaces
def remove_extra_spaces(text):
    cleaned_text = re.sub(' +', ' ', text)
    return cleaned_text.strip()


# In[29]:


# Function to expand contractions
def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text


# In[30]:


#Apply
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(remove_punctuations)
data['text'] = data['text'].apply(remove_words_with_digits)
data['text'] = data['text'].apply(remove_extra_spaces)
data['text'] = data['text'].apply(expand_contractions)


# In[31]:


data


# # Naive Bayes

# In[32]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score


# In[33]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)


# In[34]:


# Create the feature vectors
# 1) tokenization - i do not know=>["i","do","not","know"] 2)building vocabulary assign numbers to unique words 3)frequency count
#term frequency-inverse document frequency (TF-IDF).



vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# # Multinomial Naive Bayes

# In[35]:



mnb = MultinomialNB()
mnb.fit(X_train_vectorized, y_train)
mnb_predictions = mnb.predict(X_test_vectorized)
mnb_accuracy = accuracy_score(y_test, mnb_predictions)
print("Multinomial Naive Bayes Accuracy:", mnb_accuracy)


# # Bernoulli Naive Bayes

# In[36]:


bnb = BernoulliNB()
bnb.fit(X_train_vectorized, y_train)
bnb_predictions = bnb.predict(X_test_vectorized)
bnb_accuracy = accuracy_score(y_test, bnb_predictions)
print("Bernoulli Naive Bayes Accuracy:", bnb_accuracy)


# # Performance metrics

# In[38]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Calculate evaluation metrics for Multinomial Naive Bayes
print("Multinomial Naive Bayes Evaluation:")
print(classification_report(y_test, mnb_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, mnb_predictions))
print("AUC-ROC Score:", roc_auc_score(y_test, mnb.predict_proba(X_test_vectorized)[:, 1]))


# In[39]:



# Calculate evaluation metrics for Bernoulli Naive Bayes
print("Bernoulli Naive Bayes Evaluation:")
print(classification_report(y_test, bnb_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, bnb_predictions))
print("AUC-ROC Score:", roc_auc_score(y_test, bnb.predict_proba(X_test_vectorized)[:, 1]))


# # SVM

# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# In[42]:



# Train the SVM model
svm = SVC()
svm.fit(X_train_vectorized, y_train)

# Make predictions on the test set
svm_predictions = svm.predict(X_test_vectorized)


# In[43]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Evaluate the performance of the SVM model
print("SVM Evaluation:")
print(classification_report(y_test, svm_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))
print("AUC-ROC Score:", roc_auc_score(y_test, svm.decision_function(X_test_vectorized)))


# # Random forest Classifier

# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[47]:


# Train the Random Forest model
random_forest = RandomForestClassifier()
random_forest.fit(X_train_vectorized, y_train)

# Make predictions on the test set
rf_predictions = random_forest.predict(X_test_vectorized)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, rf_predictions)
precision = precision_score(y_test, rf_predictions, average='weighted')
recall = recall_score(y_test, rf_predictions, average='weighted')
f1 = f1_score(y_test, rf_predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# # DecisionTree Classifier

# In[82]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the dataset from the CSV file
data = pd.read_csv(r"C:\Users\HP\Downloads\data (2).csv", encoding='unicode_escape', header=None, names=['label', 'text'])

# Split the data into features and target variable
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the text data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create the decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[ ]:




