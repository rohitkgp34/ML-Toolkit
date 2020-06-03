# core pakages
import streamlit as st
import os

# ML/NLP/WebScrapping pakages
import json
import numpy as np
import spacy
from spacy import displacy 
import nltk
import networkx as nx
from PIL import Image
from textblob import TextBlob 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from urllib.request import urlopen


# html wrapper
HTML_WRAPPER = """<div style="overflow-x: auto; border: lpx solid #e6e9ef; border-radius: 0.25rem; padding:lrem">{}</div>"""
 
# Function for Sentence Tokenization
@st.cache
def sent_tkzn(text):
    sentences = nltk.sent_tokenize(text)
    allData = [('Token:{},\n:'.format(nltk.sent_tokenize(token)))for token in sentences]
    return allData

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData



# Function For Text Summary
@st.cache
def text_summarizer(my_text, n):
    sentences = my_text.split('.')
    
    # Extract word vectors
    import numpy as np
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    # removing punctuations, numbers and special words
    import pandas as pd
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]


    # importing stopwords
    stop_words = stopwords.words('english')

    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
      else:
        v = np.zeros((100,))
      sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
        
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    summary = []
    # Extract top 3 sentences as the summary
    for i in range(n):
      summary.append(ranked_sentences[i][1])

    ans = '.'.join(summary)
    return ans

# NER Checker 
@st.cache(allow_output_mutation=True)
def analyze_text(text):
  nlp = spacy.load('en')
  return nlp(text)


# Analyzer text from URL
@st.cache
def get_text(raw_url):
  page = urlopen(raw_url)
  soup = BeautifulSoup(page)
  fetched_text = ''.join(map(lambda p: p.text, soup.find_all('p')))
  return fetched_text







def main():

  st.title("NLPfy - Text Analytics Application")

  # Sidebar 1 (Information about NLP)
  if st.sidebar.checkbox("About NLP"):
    st.subheader("What is Natural Language Processing(NLP)?")
    st.write("NLP is a branch of artificial intelligence that deals with analyzing, understanding and generating the languages that humans use naturally in order to interface with computers in both written and spoken contexts using natural human languages instead of computer languages.")
    
    img1 = Image.open("Images/1.jpg")
    st.image(img1, width=300)
  
    st.subheader("What is Natural Language Processing good for?")
    st.write("1) Summarize blocks of text to extract the most important and central ideas while ignoring irrelevant information.")
    st.write("2) Create a chat bot, a language parsing deep learning model used by big tech gaints.")
    st.write("3) Automatically generate keyword tags from content.")
    st.write("4) Identify the type of entity extracted, such as it being a person, place, or organization using Named Entity Recognition.")
    st.write("5) Use Sentiment Analysis to identify the sentiment of a string of text, from very negative to neutral to very positive.")


  # Sidebar 2 (Implementing those Algorithms)
  if st.sidebar.checkbox("Implement NLP Algos"):
    functions = ["Sentence Tokenizer", "Word Tokenizer and Lemmentizer", "Sentiment Analysis", "Text Summarizer", "NER Checker", "NER with url"]
    choice = st.sidebar.selectbox("Choose NLP Function:",functions)

    # Sentence Tokenization
    if choice == "Sentence Tokenizer":

      st.subheader("Sentence Tokenization")

      if st.button("Description"):
        st.write("Sentence tokenization is the process of splitting text into individual sentences.")
        
        st.write("Why sentence tokenization is needed when we have the option of word tokenization? Imagine if we need to count average words per sentence, how will we calculate?") 
        st.write("For accomplishing such a task, we need both sentence tokenization as well as words to calculate the ratio. Such output serves as an important feature for machine training as the answer would be numeric.")

      message = st.text_area("Enter Text","Type Here.")
      if st.button("Check Input Text"):
        st.write(message)
      if st.button("Tokenize Text"):
        nlp_result = sent_tkzn(message)
        st.json(nlp_result)


    # Word Tokenization and Lemmantization
    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    if choice == "Word Tokenizer and Lemmentizer":
      
      st.subheader("Word Tokenizer and Lemmentizer")
      if st.button("Description"):
        st.write("Word tokenization is the process of splitting text into individual words whereas Lemmatization is the process of converting a word to its base form. Another form similar to Lemmatization is Stemming. The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.")

      message = st.text_area("Enter Text","Type Here.")
      if st.button("Tokenize  and Lemmantize Text"):
        nlp_result = text_analyzer(message)
        st.json(nlp_result)


    # Sentiment Analysis
    if choice == "Sentiment Analysis":
      
      st.subheader("Sentiment Analysis using textblob")

      if st.button("Description"):
        st.write("The sentiment function of textblob returns two properties, polarity, and subjectivity.")
        st.write("Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement. Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. Subjectivity is also a float which lies in the range of [0,1].")

      message = st.text_area("Enter Text","Type Here.")
      if st.button("Analyze text"):
        blob = TextBlob(message)
        result = blob.sentiment
        st.success(result)

    
    # Text Summarization
    if choice == "Text Summarizer":
      
      st.header("Text Summarizer")

      if st.button("Description"):
        st.write("Text summarization refers to the technique of shortening long pieces of text.")
        st.write("There are two main types of how to summarize the text in NLP:")
        st.write("1.Extraction-based summarization:")
        st.write("These methods rely on extracting several parts, such as phrases and sentences, from a piece of text and stack them together to create a summary. Therefore, identifying the right sentences for summarization is of utmost importance in an extractive method.")
        st.write("2.Abstractive-based summarization")
        st.write("These methods use advanced NLP and Deep learning techniques to generate an entirely new summary. Some parts of this summary may not even appear in the original text. The abstractive text summarization algorithms create new phrases and sentences that relay the most useful information from the original text — just like humans do. Therefore, abstraction performs better than extraction. However, the text summarization algorithms required to do abstraction are more difficult to develop; that’s why the use of extraction is still popular.")
        st.write("I've used Extraction based method.")


      message = st.text_area("Enter Text","Type Here.")
      test_msg = message
      input_lines = len(test_msg.split('.'))
      if st.button("See Input Text"):
        st.write(message)
      if st.button("Total Input Lines"):
        st.success(input_lines)
      output_lines = st.slider("Select Number of lines in Summary:", 1, 10)
      if st.button("Summarize text"):
        nlp_result = text_summarizer(message, output_lines)
        st.success(nlp_result)

    # NER Checker
    if choice == "NER Checker":
      
      st.header("Named entity recognition")
      if st.button("Description"):
        st.write("Named entity recognition (NER)is probably the first step towards information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. NER is used in many fields in Natural Language Processing (NLP), and it can help answering many real-world questions, such as:")

        st.write("1) Which companies were mentioned in the news article?")
        st.write("2) Were specified products mentioned in complaints or reviews?")
        st.write("3) Does the tweet contain the name of a person? Does the tweet contain this person’s location?")

      message = st.text_area("Enter Text", "Type Here.")
      if st.button("Analyze"):
        docx = analyze_text(message)
        html = displacy.render(docx, style='ent')
        html = html.replace("\n\n", "\n")
        st.write(html, unsafe_allow_html=True)
      
      if st.button("Extract Entities Code"):
        img2 = Image.open("Images/2.PNG")
        st.image(img2, width=800)

    # Named entity checker with url
    if choice == "NER with url":

      st.subheader("Analyze text from url")
      raw_url = st.text_input("Enter URL", "Type Here.")
      text_length = st.slider("Lenght to preview", 50, 100)
      if st.button("Extract"):
        if raw_url != "Type Here.":
          result = get_text(raw_url)
          #text_length = len(result)
          #preview_length = st.slider("Lenght to preview", 1, text_length)
          lft = len(result)
          lst = round(len(result)/text_length)
          summary_docx = text_summarizer(result, 5)
          docx = analyze_text(summary_docx)
          html = displacy.render(docx, style='ent')
          html = html.replace("\n\n", "\n")
          st.write(html, unsafe_allow_html=True)



  # Sidebar 3 (Credits)
  st.sidebar.header("About")
  st.sidebar.text("By Rohit Mahajan")
  st.sidebar.text("3rd Year UG at IIT-Kharagpur")
  st.sidebar.text("rohitmahajan2810@gmail.com")


if __name__ == "__main__":
  main()