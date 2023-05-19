from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import openpyxl

st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Enter Text Here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity:', round(blob.sentiment.polarity,2))
        st.write('Sentiment :', round(blob.sentiment.subjectivity,2))


    pre = st.text_input('Clean Text:')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True ))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload File')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_csv(upl)
        df.drop(['id','url','timestamp','replies','retweets','quotes','likes'], axis=1, inplace=True)
        df['score'] = df['tweet'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head())


        @st.cache_data
        def convert_df(df):

            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )