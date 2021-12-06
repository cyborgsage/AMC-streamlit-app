import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn import pipeline
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

header = st.container()
dataset = st.container()
model_training = st.container()

with header:
    st.title('Welcome to AMC Comment Sentiment Analysis')
    st.text('This application predicts the sentiment of AMC-related comments.')

with dataset:
    st.header('AMC comment dataset')
    st.text('I scraped this dataset from r/wallstreetbets with PushShift API')
    st.sidebar.title('AMC Comment Sentiment Analysis')
    st.sidebar.markdown('This analyzer will classify comments into bullish (positive), bearish (negative), or neutral')
    data=pd.read_csv('labeled_amc.csv')
    if st.checkbox('Show Data'):
        st.write(data.head())

    st.sidebar.header('Comment Analyzer')
    comments = st.sidebar.radio('Sentiment Type',('Positive','Neutral','Negative'))
    st.write(data.query('labels==@comments')[['body']].sample(1).iat[0,0])
    st.write(data.query('labels==@comments')[['body']].sample(1).iat[0,0])

    sentiment=data['labels'].value_counts()
    sentiment=pd.DataFrame({'Sentiment':sentiment.index, 'Comments':sentiment.values})
    st.text('Sentiment Count')
    fig = px.bar(sentiment, x='Sentiment', y='Comments', color='Comments',height=500)
    st.plotly_chart(fig)

with model_training:
    st.header('Time to train the model!')
    st.text('Choose the hyperparameters of the model and see how the performance changes')

    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=200, value=20, step=10)
    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100, 500, 1000, 2000, 3000, 10000], index=0)
    input_feature = sel_col.text_input('Which feature should be used as the input feature?','labels')

    X = data['body']
    y = data['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def comment_cleaner(comment):
        common_symbols = ['gt','amp','x200B','==&gt;','&gt','&amp;','#x200B;','\n']
        punctuation = set(string.punctuation)
        to_keep = ['!','$','%','@']
        punctuation.difference_update(to_keep) #Keep exclamations and other contextual symbols
        comment = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', comment) #Remove URLs and links
        tokenizer = TweetTokenizer()
        comment = tokenizer.tokenize(comment)
        comment = [word for word in comment if word not in common_symbols] #Remove reoccuring symbol phrases
        comment = [word for word in comment if word not in punctuation] #Remove punctuation
        comment = pos_tag(comment)
        comment = [(word[0], get_wordnet_pos(word[1])) for word in comment]
        lemmatizer = WordNetLemmatizer() 
        comment = [lemmatizer.lemmatize(word[0], word[1]) for word in comment]
        return comment

    rf_pipe = pipeline.Pipeline(steps=[('pre', TfidfVectorizer(tokenizer=comment_cleaner, max_features=1200)),
                                       ('smote', SMOTE(random_state=42)),
                                       ('rf', RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=4,
                                                       min_samples_leaf=1,min_samples_split=2))])
    rf_pipe.fit(X_train, y_train)
    y_pred = rf_pipe.predict(X_test)
    disp_col.markdown('Training Score')
    disp_col.write(rf_pipe.score(X_train, y_train))
    disp_col.markdown('Test Score')
    disp_col.write(accuracy_score(y_test, y_pred))


