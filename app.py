# import libraries 
import pandas as pd
import streamlit as st 
import pickle 
import re
import string
from pathlib import Path

# load our data frame samples
from pathlib import Path

fake_path = Path.cwd() / "Sample Data/fake_samples.csv"
fake_sample = pd.read_csv(r'/Users/sakethreddy/Fake (1).csv', encoding='latin-1',on_bad_lines='skip')

true_path = Path.cwd() / "Sample Data/true_samples.csv"
true_sample = pd.read_csv(r'/Users/sakethreddy/True (1).csv', encoding='latin-1',on_bad_lines='skip')


# load our saved models using pickle 

tree_path = Path("/Users/sakethreddy/decision_tree.sav")
decision_tree = pickle.load(open(tree_path, "rb"))

# Load the TF-IDF vectorizer
vectorizer_path = Path("/Users/sakethreddy/tfid_algo.sav")
vectorizer = pickle.load(open(vectorizer_path, "rb"))


# create a function to clean text
def wordopt(text):
    text = text.lower() # lower case 
    text = re.sub('\[.*?\]','',text) # remove anything with and within brackets
    text = re.sub('\\W',' ',text) # removes any character not a letter, digit, or underscore
    text = re.sub('https?://\S+|www\.\S+','',text) # removes any links starting with https
    text = re.sub('<.*?>+','', text) # removes anything with and within < >
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # removes any string with % in it 
    text = re.sub('\n','',text) # remove next lines
    text = re.sub('\w*\d\w*','', text) # removes any string that contains atleast a digit with zero or more characters
    return text
  

# prediction function 
def news_prediction(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_tfidf_test = vectorizer.transform(new_x_test)
    pred_dt = decision_tree.predict(new_tfidf_test)
    
    if (pred_dt[0] == 0):
      return "This is Fake News!"
    else:
      return "The News seems to be True!"
    
    
    
def main():
  
  # write our title 
  st.title("Real or Fake News Prediction System")
  
  st.write("""In today's era of technology and social media, determining the validity of news has become increasingly challenging. Fake news can manipulate perceptions of reality, influence politics, and promote false advertising, stirring social conflict. This misinformation leads to growing mistrust and confusion. This app predicts whether a news article is fake or real. Just copy and paste the text into the box below and click the predict button.""")
  
  st.write("""This app predicts if a news article contains Fake News or not. Just copy and paste the text into the following box
            and click on the predict button.""")
  
  st.write("""## Input your News Article down below: """)
  
  user_text = st.text_area(':blue[Text to Analyze]',  '''Pope Francis used his annual Christmas Day message to rebuke Donald Trump without even mentioning his name. The Pope delivered his message just days after members of the United Nations condemned Trump s move to recognize Jerusalem as the capital of Israel. The Pontiff prayed on Monday for the  peaceful coexistence of two states within mutually agreed and internationally recognized borders. We see Jesus in the children of the Middle East who continue to suffer because of growing tensions between Israelis and Palestinians,  Francis said.  On this festive day, let us ask the Lord for peace for Jerusalem and for all the Holy Land. Let us pray that the will to resume dialogue may prevail between the parties and that a negotiated solution can finally be reached. The Pope went on to plead for acceptance of refugees who have been forced from their homes, and that is an issue Trump continues to fight against. Francis used Jesus for which there was  no place in the inn  as an analogy. Today, as the winds of war are blowing in our world and an outdated model of development continues to produce human, societal and environmental decline, Christmas invites us to focus on the sign of the Child and to recognize him in the faces of little children, especially those for whom, like Jesus,  there is no place in the inn,  he said. Jesus knows well the pain of not being welcomed and how hard it is not to have a place to lay one s head,  he added.  May our hearts not be closed as they were in the homes of Bethlehem. The Pope said that Mary and Joseph were immigrants who struggled to find a safe place to stay in Bethlehem. They had to leave their people, their home, and their land,  Francis said.  This was no comfortable or easy journey for a young couple about to have a child.   At heart, they were full of hope and expectation because of the child about to be born; yet their steps were weighed down by the uncertainties and dangers that attend those who have to leave their home behind. So many other footsteps are hidden in the footsteps of Joseph and Mary,  Francis said Sunday. We see the tracks of entire families forced to set out in our own day. We see the tracks of millions of persons who do not choose to go away, but driven from their land, leave behind their dear ones. Amen to that.Photo by Christopher Furlong/Getty Images.''', height = 350)
  
  if st.button("Article Analysis Result"):
    news_pred = news_prediction(user_text)
    
    if (news_pred == "This is Fake News!"):
      st.error(news_pred, icon="🚨")
    else:
      st.success(news_pred)
      st.balloons()
  
  
if __name__ == "__main__":
  main()
