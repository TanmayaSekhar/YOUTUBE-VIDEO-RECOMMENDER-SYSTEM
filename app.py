# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:34:23 2021

@author: SUBRAT PATRA
"""

import streamlit as st 
import streamlit.components.v1 as stc 
from spacy.lang.en.stop_words import STOP_WORDS
from gensim import corpora
from operator import itemgetter
from gensim.similarities import MatrixSimilarity
import string
import spacy
import re
import gensim 

# Load EDA
import pandas as pd 
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

data = pd.read_csv('D:\\Study\\Projects\\Youtube Video Recommendation\\Final Data\\YouTubeData.csv')
data1=data.drop(['Unnamed: 0','v_id','Keyword','publishedDate','viewCount','likeCount','dislikeCount','commentCount'],axis=1)
data1["v_title1"]=data.v_title
data1["v_description1"]=data.v_description

spacy_nlp = spacy.load('en_core_web_sm')

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def spacy_tokenizer(sentence):
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens

data1['v_title1'] = data1['v_title1'].map(lambda x: spacy_tokenizer(x))
v_title1=data1["v_title1"]

word = corpora.Dictionary(v_title1)

corpus = [word.doc2bow(desc) for desc in v_title1 ]

word_frequencies = [[(word[id], frequency) for id, frequency in line] for line in corpus[0:2357]]

tfidf_model = gensim.models.TfidfModel(corpus, id2word=word)
lsi_model = gensim.models.LsiModel(tfidf_model[corpus], id2word=word, num_topics=300)

gensim.corpora.MmCorpus.serialize('tfidf_model_mm', tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('lsi_model_mm',lsi_model[tfidf_model[corpus]])

#Load the indexed corpus
tfidf_corpus = gensim.corpora.MmCorpus('tfidf_model_mm')
lsi_corpus = gensim.corpora.MmCorpus('lsi_model_mm')

video_index = MatrixSimilarity(lsi_corpus, num_features = lsi_corpus.num_terms)

def video_rec(search_term):

    query_bow = word.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = tfidf_model[query_bow]
    query_lsi = lsi_model[query_tfidf]

    video_index.num_best = 10

    video_list = video_index[query_lsi]

    video_list.sort(key=itemgetter(1), reverse=True)
    video_names = []

    for j, video in enumerate(video_list):

        video_names.append (
            {
                'video_link':"https://www.youtube.com/watch?v=" + data['v_id'][video[0]],
                'video_Title': data['v_title'][video[0]],
                'video_view_count':data['viewCount'][video[0]],
                'video_like_count':data['likeCount'][video[0]],
                'video_dislike_count':data['dislikeCount'][video[0]],
                'video_comment_count':data['commentCount'][video[0]]
            }

        )
        if j == (video_index.num_best-1):
            break

    return pd.DataFrame(video_names, columns=['video_link','video_Title','video_view_count','video_like_count','video_dislike_count','video_comment_count'])



#Create the Main title of web page 
st.title("You Tube Video Recommender Engine")


#create the color background & subtitle 
html_temp = """
<div style="background-color:tomato;padding:9px">
<h1 style="color:white;text-align:center;">Key Word Based Video Recommendation</h>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)


#st.markdown(page_bg_img, unsafe_allow_html=True)

#Create the nevigator or sidebar

nav = st.sidebar.radio("Navigation",["Home","Recommendation","None"])


#Create condition according desired output
if nav == "Home":
    st.write("Welcome To Content Based Youtube Video Recommender System")
    
if nav == "Recommendation":
    st.subheader("Recommend Courses")
	#cosine_sim_mat = vectorize_text_to_cosine_mat(data['v_title'])
search_term = st.text_input("Search")
	#num_of_rec = st.sidebar.number_input("Number",4,30,7)
if st.button("Recommend"):
		if search_term is not None:
			try:
				results = video_rec(search_term)
				with st.beta_expander("Results as JSON"):
					results_json = results.to_dict('index')
					st.write(results_json)
                    
				#stc.html(results_json) 

				
			except:
				results= "Not Found"
				st.warning(results)
				#st.info("Suggested Options include")
				#result_df = search_term_if_not_found(search_term,df)
				#st.dataframe(result_df)

    
if nav == "None":
    #st.subheader("About")
	st.write("Built with Streamlit & Pandas")