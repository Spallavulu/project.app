from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext


st.header('Real or Fake News Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >= 0.03:
            return 'Positive'
        elif x <= -0.03:
            return 'Negative'
        else:
            return 'Neutral'

#
    if upl:
        df = pd.read_csv(upl,encoding='latin-1',names=range(8), header=None)
        del df[0]
        df.columns = ['tweets', 'other_col1', 'other_col2', 'other_col3', 'other_col4', 'other_col5', 'other_col6']  # Rename columns if needed
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('latin-1')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )



