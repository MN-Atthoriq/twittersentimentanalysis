#import library
import streamlit as st
import pickle
import time
import pandas as pd
import numpy as np
import preprocess_kgptalkie as ps
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

#set up layout -> wide layout
st.set_page_config(layout="wide")

#create header text
st.title(':blue[Twitter] Sentiment :blue[Analysis] & :blue[Prediction]')
st.markdown(':blue[Explore] many :blue[relationships] in sentiment of various tweet and :blue[predict] sentiment of :blue[future] tweet!')

#create tabs
tab1, tab2, tab3 = st.tabs([":clipboard: Data", ":chart: Visualize", ":mag: Prediction"])

with tab1: #dataset
    #load dataset and data cleaning
    df = pd.read_csv('twitter.csv', header=None, index_col=0)
    df = df[[2,3]].reset_index(drop=True)
    df.columns = ['sentiment', 'text']
    df.dropna(inplace=True)
    df = df[df['text'].apply(len)>5]

    #make description about dataset
    with st.expander("See description"): 
        st.write('This dataset have ', df.shape[1], ' rows and ', df.shape[0], ' columns with various sentiments:')
        st.write('- Positive:   ', df['sentiment'].value_counts()[1], ' data /', round(df['sentiment'].value_counts()[1]/df.shape[0]*100,2),'% of dataset')
        st.write('- Negative:   ', df['sentiment'].value_counts()[0], ' data /', round(df['sentiment'].value_counts()[0]/df.shape[0]*100,2),'% of dataset')
        st.write('- Neutral:   ', df['sentiment'].value_counts()[2], ' data /', round(df['sentiment'].value_counts()[2]/df.shape[0]*100,2),'% of dataset')
        st.write('- Irrelevant:   ', df['sentiment'].value_counts()[3], ' data /', round(df['sentiment'].value_counts()[3]/df.shape[0]*100,2),'% of dataset')
    
    #show data
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     "sentiment": "Sentiment",
                     "text": "Tweet"
                 })
    
    #button for dataset reference 
    st.link_button('Twitter Sentiment Dataset', 'https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis?resource=download', type='primary')

with tab2: #visualize
    #calculate some characteristic of dataset and renaming columns label
    df = ps.get_basic_features(df) 
    col = ['sentiment', 'text', 'Character Counts', 'Word Counts', 'Average Word Length','Stopwords Counts', 'Hashtag Counts', 'Mention Counts', ' Digit Counts', 'Uppercase Counts']
    df.columns = col

    col1, col2 = st.columns([1,3]) #make two columns layout

    with col1:
        #input plot
        plot = st.selectbox('What plot do you want to know?',('Kernel Density Plot', 'Wordcloud'))

        if plot == 'Wordcloud':
            #input sentiment
            wc = st.selectbox('What sentiment do you want to plot?',('Positive','Negative','Neutral','Irrelevant'))

        else:
            #input characteristic
            kde = st.selectbox('What characteristics do you want to plot?',
                               ('Character Counts', 'Word Counts', 'Average Word Length','Stopwords Counts', 'Hashtag Counts',
                                'Mention Counts', ' Digit Counts', 'Uppercase Counts'))
    
    with col2:
        if plot == 'Wordcloud': #make wordcloud
            stopwords = set(STOPWORDS)
            fig = plt.figure(figsize=(12,6))
            data = df[df['sentiment']==wc]['text']
            wordcloud = WordCloud(background_color='white', stopwords=stopwords,
                          max_words=200, max_font_size=40, scale=5).generate(str(df['text']))
            plt.imshow(wordcloud)
            plt.xticks([])
            plt.yticks([])
            plt.title(str(wc)+' Sentiment Wordcloud', fontdict={'fontsize': 20})
            plt.show()
            st.pyplot(fig)

        else: #make kde plot
            fig = plt.figure(figsize=(12,6))
            sns.kdeplot(data=df, x=kde, hue='sentiment', fill=True)
            plt.tight_layout()
            plt.title(str(kde)+' KDE Plot', fontdict={'fontsize': 20})
            plt.show()
            st.pyplot(fig)
    
with tab3: #prediction
    #load model
    model = pickle.load(open('twitter.pkl', 'rb'))

    #input
    tweet = st.text_input('A', placeholder='Enter your tweet!', label_visibility='collapsed')
    submit = st.button('Predict!', type='primary')

    if submit:
        #prediction and calculate time needed for predict
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()

        #output time needed for prediction and sentimen prediction result
        st.write('Prediction time taken: ', round(end - start, 2), ' seconds')
        if prediction[0] == 'Positive':
            st.write('Sentiment Prediction is: ')
            st.header(':green[Positive] :smile:')

        elif prediction[0] == 'Negative':
            st.write('Sentiment Prediction is: ')
            st.header(':red[Negative] :rage:')

        elif prediction[0] == 'Neutral':
            st.write('Sentiment Prediction is: ')
            st.header(':white[Neutral] :neutral_face:')

        else: #irrelevant
            st.write('Sentiment Prediction is: ')
            st.header(':blue[Irrelevant] :frowning:')
        
        #description about model after user predict something
        with st.expander("See explanation of the model prediction"):
            st.markdown('This Twitter sentiment prediction uses a machine learning model of a :blue[random forest classifier].')
            st.markdown('This model is :blue[built] on the Twitter dataset used in this dashboard.')
            st.markdown('This model has a :blue[high accuracy rate], around 9 out of 10 predictions made are :blue[correct] predictions.')
            st.link_button('Learn More About Random Forest', 'https://en.wikipedia.org/wiki/Random_forest', type='primary')