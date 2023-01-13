#!/usr/bin/env python
# coding: utf-8

# # RoBERTa (BERT) vs Logistic Regression
# ## Overview
# This project will be showcasing the steps to build two different emotion detection NLP models (using RoBERTa and Logistic Regression, as the title suggests). We will also be looking at factors such as the spread of the dataset, time for training, and the accuracy of both models. 
# 
# ## Prerequisites
# If you want to run this project on your own machine, make sure to have the following items installed:
# \
# Python and Jupyter Notebook, Tensorflow, and all Python packages imported below. 
# \
# The dataset files ('test.txt', 'train.txt', 'val.txt'), which can be found at the link in the "Sources" section at the end of this notebook. Take these files and put them in a subdirectory (called 'data'), relative of the directory of this notebook. 

# ### Model 1: Logistic Regression
# We start by building the simpler model, using logistic regression. Before beginning, we import the needed packages for this first part of making the model. Then, we start by importing the dataset into Pandas DataFrames and combining them into one DataFrame, and then preprocessing the text by running some NeatText functions. Then, we'll use scikit-learn to apply Logisitic Regression. 

# In[1]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neattext.functions as nfx
import pickle


# In[2]:


# Loading Dataset
df1 = pd.read_csv('./data/test.txt', names=['text', 'emotion'], sep=';')
df2 = pd.read_csv('./data/train.txt', names=['text', 'emotion'], sep=';')
df3 = pd.read_csv('./data/val.txt', names=['text', 'emotion'], sep=';')


# In[3]:


df = df1.append(df2,ignore_index=True)


# In[4]:


df = df.append(df3,ignore_index=True)


# In[5]:


# Text Cleaning
df['clean_text'] = df['text'].apply(nfx.remove_stopwords)


# In[6]:


df['clean_text'] = df['clean_text'].apply(nfx.remove_userhandles)


# In[7]:


df['clean_text'] = df['clean_text'].apply(nfx.remove_punctuations)


# In[8]:


# Importing ML Model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[9]:


# Splitting Dataset
from sklearn.model_selection import train_test_split


# In[10]:


x_features = df['clean_text']
y_labels = df['emotion']


# In[11]:


# Vectorizer
cv = CountVectorizer()
X = cv.fit_transform(x_features)


# In[39]:


x_train,x_test_lr,y_train,y_test_lr = train_test_split(X,y_labels,test_size=0.3,random_state=42)


# In[40]:


lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)


# In[41]:


lr_model.score(x_test_lr,y_test_lr)


# In[15]:


# Make a prediction
sample_text = ["I'm a bit mad"]


# In[16]:


vect = cv.transform(sample_text).toarray()


# In[17]:


lr_model.predict(vect)


# In[18]:


# Saving model and vectorizer
pickle.dump(lr_model, open('lr_model.sav', 'wb'))
pickle.dump(cv, open('vectorizer.pickle', 'wb'))


# ## Model 2: RoBERTa
# We will now following the same series of steps, with changes in our approach, to build a model based off of RoBERTa. 

# In[19]:


# Imports
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import neattext.functions as nfx
from sklearn.model_selection import train_test_split


# In[20]:


train_data = pd.read_csv('./data/train.txt', names=['text', 'emotion'], sep=';')
val_data = pd.read_csv('./data/val.txt', names=['text', 'emotion'], sep=';')
test_data = pd.read_csv('./data/test.txt', names=['text', 'emotion'], sep=';')


# In[21]:


# Checking for null values in the dataset
train_data['emotion'].isnull().sum()
val_data['emotion'].isnull().sum()
test_data['emotion'].isnull().sum()


# In[22]:


# Pre-Processing
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^A-z]', ' ', text)
    negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except', 'even though', 'yet']
    stop_words = [word for word in stop_words if word not in negative]
    preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in text.split() if temp not in stop_words and temp[0] != "["]
    return ' '.join([x for x in preprocessed_tokens]).strip()


# In[23]:


train_data['text'] = train_data['text'].apply(lambda x: preprocess(x))
val_data['text'] = val_data['text'].apply(lambda x: preprocess(x))
test_data['text'] = test_data['text'].apply(lambda x: preprocess(x))


# In[24]:


# Adding repetition to all classes except the highest frequency class
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
train_x, train_y = ros.fit_resample(np.array(train_data['text']).reshape(-1,1), np.array(train_data['emotion']).reshape(-1,1))
train = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text', 'emotion'])


# In[25]:


# Test
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
train_x, train_y = ros.fit_resample(np.array(train_data['text']).reshape(-1, 1), np.array(train_data['emotion']).reshape(-1, 1))
train = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text', 'emotion'])


# In[26]:


# Adding OneHotEncoder
from sklearn import preprocessing
le = preprocessing.OneHotEncoder()
y_train = le.fit_transform(np.array(train['emotion']).reshape(-1,1)).toarray()
y_test = le.fit_transform(np.array(test_data['emotion']).reshape(-1,1)).toarray()
y_val = le.fit_transform(np.array(val_data['emotion']).reshape(-1,1)).toarray()
y_train


# In[27]:


# Encoding
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


# In[28]:


def roberta_encode(data,maximum_length):
    input_ids = []
    attention_masks = []
    
    for i in range(len(data.text)):
        encoded = tokenizer.encode_plus(
            data.text[i],
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


# In[29]:


max_len = max([len(x.split()) for x in train_data['text']])
train_input_ids,train_attention_masks = roberta_encode(train,max_len)
test_input_ids,test_attention_masks = roberta_encode(test_data,max_len)
val_input_ids,val_attention_masks = roberta_encode(val_data,max_len)
type(test_input_ids)


# In[30]:


# Creating Model
def create_model(bert_model, max_len):
    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    output = bert_model([input_ids,attention_masks])
    output = output[1]
    
    output = tf.keras.layers.Dense(6,activation='softmax')(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    #model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(Adam(learning_rate=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# In[31]:


from transformers import TFRobertaModel
roberta_model = TFRobertaModel.from_pretrained('roberta-base')


# In[32]:


model = create_model(roberta_model, max_len)
model.summary()


# In[33]:


# Training Model (I ran 1 epoch for speed reasons, but increase this to 4+ epochs for higher accuracy)
history = model.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_val),
                   epochs=4,batch_size=100)


# In[34]:


# Saving Weights
model.save_weights('roberta_emotion_1')


# ## Comparisons
# Now that both models have been created, it is time to compare the two! We will start by looking at training time, then graphing the spread of emotions in the dataset, then plotting accuracy graphs of the model, and finally calculating the F1 scores of the two models and comparing their accuracies. 
# \
# One factor to consider is training time and resource consumption of the two models. When training the logistic regression model, it only took a matter of seconds, while training four epochs of the RoBERTa-based model took about 26 minutes on my machine. While this is still not a very long time, when using larger datasets, which RoBERTa requires to be more accurate, it can take dozens of hours to train, while logistic regression may just take a higher number of seconds.
# \
# Similarly, when applying the model, logistic regression is much faster and has a lighter load on the machine, while RoBERTa takes a bit of time. Thus, when developing an application that needs to call on the model frequently and return the result to the user, logistic regression would likely be a better choice. 

# In[35]:


# Plotting graph of dataset spread (number of occurrences of each emotion)
plt.figure(figsize=(15,10))
sns.countplot(x='emotion',data=df)
plt.show


# In[36]:


# Accuracy and F1 Score
result = model.predict([test_input_ids,test_attention_masks])
y_pred = np.zeros_like(result)
y_pred[np.arange(len(result)),result.argmax(1)] = 1


# In[42]:


from sklearn.metrics import accuracy_score, f1_score
accuracy_lr = lr_model.score(x_test_lr,y_test_lr)
accuracy_roberta = accuracy_score(y_test,y_pred)
print('Accuracy of Logistic Regression model', accuracy_lr)
print('Accuracy of RoBERTa model', accuracy_roberta)
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 Score of RoBERTA model:', f1)


# ## Conclusion
# - RoBERTa is a more complex model, that takes longer to train and use, but results in getting more accurate results. 
# - Logistic Regression is faster in both training and applying, but may be a bit less accurate as a tradeoff. 
# - Both models have their benefits and drawbacks, and different usecases may govern which one to go with. If you need a really accurate model that you are periodically going to call, RoBERTa is the one to go with. If you need something fast, logistic regression may be the way to go. 
# 
# Sources:
# \
# Dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
# \
# RoBERTa Model: https://www.kaggle.com/code/dhruv1234/emotion-classification-roberta/notebook
