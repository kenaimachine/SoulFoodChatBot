#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd  
##import numpy as np  
#import matplotlib.pyplot as plt  
#import seaborn as seabornInstance 

##from sklearn.model_selection import train_test_split 
##from sklearn import neighbors
##from sklearn.metrics import mean_squared_error
##from sklearn.metrics import mean_absolute_error
##from sklearn.metrics import accuracy_score
##from sklearn import metrics
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
##import gensim
##from gensim.test.utils import get_tmpfile

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import ast
import re

import sqlite3
from sqlite3 import Error

from flask import Flask, request, jsonify, render_template
import os
#import dialogflow
##import requests
##import json



import pickle
#%matplotlib inline
#nltk.download('stopwords')


# In[105]:


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)   # get the incoming JSON structure
    action = data['queryResult']['action'] # get the action name associated with the matched intent
    
    if (action == 'get_task'):
        return get_tasks(data)
    elif (action == 'get_task_simple'):
        return get_tasks_quick(data)
    else: None
    


# In[130]:


def get_tasks_quick(data):
    outputContextData= data['queryResult']['outputContexts']
    sentiment_text, affinity_modifiers, like_text, dislike_text=extract_data_simple(outputContextData)
    
    message_list= [sentiment_text, affinity_modifiers, like_text, dislike_text]
    
    rm=Recommender_Model()
    reply=rm.query_recommendation_model(message_list)
    
    #for testing, comment out jsonify(reply)
    return jsonify(reply)
    #print (reply)


# In[131]:


#this function extract the data from the fulfilment request.
def extract_data_simple(outputContextData):
    for i in range(len(outputContextData)):
            
        if 'ctx_like_text' in outputContextData[i]['name']:
            like_text=outputContextData[i]['parameters']['like_text']
            
        if 'ctx_dislike_text' in outputContextData[i]['name']:
            dislike_text=outputContextData[i]['parameters']['dislike_text']
                
            
    sentiment_text='neutral, neutral, neutral' # return sentiment as neutral
    
    
    affinity_modifiers=affinity_tester(5,5) #make the affinity neutral
    
    
    return sentiment_text, affinity_modifiers, like_text, dislike_text


# In[107]:


def get_tasks(data):
    outputContextData= data['queryResult']['outputContexts']
    sentiment_text, affinity_modifiers, like_text, dislike_text=extract_data(outputContextData)
    
    message_list= [sentiment_text, affinity_modifiers, like_text, dislike_text]
    
    rm=Recommender_Model()
    reply=rm.query_recommendation_model(message_list)
    
    #for testing, comment out jsonify(reply)
    return jsonify(reply)
    #print (reply)


# In[96]:


#this function extract the data from the fulfilment request.
def extract_data(outputContextData):
    for i in range(len(outputContextData)):
        if 'ctx_sentiment_text' in outputContextData[i]['name']:
            sentiment_text_1=outputContextData[i]['parameters']['sentiment_text_1']
            sentiment_text_2=outputContextData[i]['parameters']['sentiment_text_2']
            sentiment_text_3=outputContextData[i]['parameters']['sentiment_text_3']
            
        if 'ctx_like_text' in outputContextData[i]['name']:
            like_text=outputContextData[i]['parameters']['like_text']
            
        if 'ctx_dislike_text' in outputContextData[i]['name']:
            dislike_text=outputContextData[i]['parameters']['dislike_text']
                
        if 'ctx_affinity_test' in outputContextData[i]['name']:
            affinity_test_1=outputContextData[i]['parameters']['affinity_test_1']
            affinity_test_2=outputContextData[i]['parameters']['affinity_test_2']
    
    sentiment_text=','.join([sentiment_text_1,sentiment_text_2,sentiment_text_3])
    
    
    affinity_modifiers=affinity_tester(affinity_test_1,affinity_test_2)
    
    
    return sentiment_text, affinity_modifiers, like_text, dislike_text


# In[2]:


#this test implement the user's affinity to text review or composite rating. And select the modifier accordingly. 
def affinity_tester(affinity_test_1,affinity_test_2):
    tally=0
    try: 
        if int(affinity_test_1)==1:  #didn't deviate from user rating
            tally+=0
        elif int(affinity_test_1)!=1: #deviate from user rating
            tally+=1
        elif int(affinity_test_2)==5: #didn't deviate from user rating
            tally+=0
        else  :#int(affinity_test_2)!=1:#deviate from user rating.-> int(affinity_test_2)!=1
            tally+=1
    except:    
        tally=tally #for invalied entries such as characters.
    
    if tally ==2:
        return {'cs_ed_modifier': 0.0, 'semantic_modifier': 2.0} #biased towards semantic scoring
    elif tally==0:
        return {'cs_ed_modifier': 1.2, 'semantic_modifier': 0.8} #biased towards composite rating
    else:
        return {'cs_ed_modifier': 0.5, 'semantic_modifier': 1.5} #default neutral affinity.


# In[ ]:


import random, os
from PIL import Image
def get_random_pictures():

    a='https://aks7fg.ch.files.1drv.com/y4mr3c5LBrF9OF_Ngijwtyz5hi2gwNlVX5-67Ac-hOKb4tDuNadYTr1q4eLITKlgrFLGHJ3uXeKCzpeyt9lgWjIY2sfmORdtSgiT_pf6NLCunpHllzvTWvEXPlm0Z3l93MezBwYqNH5v7s5UjKTxgQ98iaPZtTKzl09nok5BTekeLRjAYDoxxlXF_MMymKQQNCvtwkN_KHFBGwG8sf85K6zsg?width=1080&height=1080&cropmode=none'
    b='https://q6o7fg.ch.files.1drv.com/y4mOPATTNA1zIv9zF0ByGEqZT-30v0b12EW8sHOQifNwmYH7xn6C0RPo9GL8Drydtp4EirwVNW68IU_VVHDxw4q5W6aYeqe0QYjOspkydDaUjs_wxeLXVkBeKq_LPrcuQNbY5ZwGlRFsBo3oPRu54p8puq8yBR3Ge7I5b6q5WwT2ABfbzs76J8Bgl-dWCpWHzLoBh3TJYA-hnwoOfv4L1zRmQ?width=1080&height=1080&cropmode=none'
    c='https://aqs7fg.ch.files.1drv.com/y4mJMBHiafy6ZZjoaBIIp1Fc-AsZR_2p6Ldg1KW6f6x9XvxZiQR4Xqdw4CMdXKSzeGsI3Ks6G40Y5_gZopZNlhb3yKVS11xmHwSpxZlmTIqxizbS5Na3vWt3gDkHelgidE5pYPeqZnloTORqdOoCHUMPJqTKw1c3EuW-OcO3lykQyTakyv6jBD8n6NGFtDw8b-z82gpabqzHVQjtAW8uVWXDw?width=1080&height=1080&cropmode=none'
    d='https://bks7fg.ch.files.1drv.com/y4mHvzWl2Tr4VqHuw4BnOF47EvmYqu0_5f2ZCEjCsFNHmoB41egvRW0VTjy1N76XSANUSDdswW8N43HZdJs3EKJHw3K1UwFRQbpjiS1wdBHZj6XM-kTQ3MaULupZQPmpoIcVvn0ANeNRscXggcjI1Pgu28OqBYzreH7IiKrM9I3JHymF-M3rzGQ4Gm5l3C8woGJxyQKxjse44tf_TJDEQ-CPA?width=1080&height=1080&cropmode=none'
    e='https://a6s7fg.ch.files.1drv.com/y4muOm3DZ5D-lczBAj1-VFi__e-_snfL6n72-jqxiFaVxCH4gIr5ZlwzDLhSxcEW5YMr-Z3jsL_CnvlxSRLpgDzHpfUXqtvyQItXpab5Otb1p5mx_vKoAFfUY6OlX2sidGVtE8FQrw7SUo4Qn_g72UyfUdTMefgQEhs5gcgUN3jXUrwS_eRa3NwCkPXtKt0Q8AxKyw2k3L7KUt9IBXJXgZQSQ?width=1080&height=1080&cropmode=none'
    f='https://aas7fg.ch.files.1drv.com/y4mO_vjBM7VHLOqMOjsgu9EpHMlgB3WBdhCAhv0fsSTon56zDJSxeEsJQwp6-AWkRCj3VFidVEm568ILr1lCguLARJMlM30LwHVJqUtnj6QH34kbntmvUNzG_0zr7i3Fogsi3Kyr02o8p8zg1oSf70Lg2dNE5T75VqHMH2KxRrjgv50axPhlkrx8apiLCgjmg8xgN5dw5QjMz0mEDSho_Doxw?width=1080&height=1080&cropmode=none'

    photo=[a,b,c,d,e,f]
    return random.choice(photo)


# #randomly select photos of restaurants because actual database for real photos will be too large for this project. 
# import random, os
# from PIL import Image
# def get_random_pictures():
#     #photo=random.choice([x for x in os.listdir("/users/KennethMacBookPro/Downloads/Capstone_Project/images")
#     #           if os.path.isfile(os.path.join("/users/KennethMacBookPro/Downloads/Capstone_Project/images", x))])
#     
#     #photopath="/users/KennethMacBookPro/Downloads/Capstone_Project/images/"+photo
#     photo=random.choice([x for x in os.listdir("https://www.pythonanywhere.com/user/kenaimachine/files/home/kenaimachine/soulfood/images")
#                if os.path.isfile(os.path.join("https://www.pythonanywhere.com/user/kenaimachine/files/home/kenaimachine/soulfood/images", x))])
#     
#     photopath="https://www.pythonanywhere.com/user/kenaimachine/files/home/kenaimachine/soulfood/images"+photo
#     
#     
#     
#     #print (photopath)
# 
#     #Image.open("/users/KennethMacBookPro/Downloads/Capstone Project/images/"+photo)

# In[ ]:





# In[6]:


class Recommender_Model:
    def __init__(self):
        self.df_restaurant_LA=pd.read_csv('/home/kenaimachine/soulfood/df_restaurant_LA_buildfinal.csv',low_memory=False)
        #self.df_restaurantDetails=pd.read_csv('df_restaurantDetails.csv',low_memory=False)
        self.loaded_rfc = pickle.load(open('/home/kenaimachine/soulfood/rfr_model_result.pkl', 'rb'))
        self.doc2vecModel = pickle.load(open('/home/kenaimachine/soulfood/doc2vec_model.pkl', 'rb'))
        self.doc2vec_featureMatrix = pickle.load(open('/home/kenaimachine/soulfood/doctovec_embeddings.pkl', 'rb'))

                
        self.df_LA_scaled=pd.DataFrame()
        self.fullUserScore=pd.DataFrame()
        self.analyser=SentimentIntensityAnalyzer()
        
        self.sentiment_cs_index=pd.DataFrame()
        self.sentiment_ed_index=pd.DataFrame()
        
    def get_df_LA_scaled(self):
        #scaling the relevant columns' data to reduce the the effect or some ratings skewing the cosine similiarity calculations.
        selected_col=['stars_x','useful','stars_y_userMean','stars_y_userRating']
        df_scaled=minmax_scale(self.df_restaurant_LA[selected_col])
        
        df_scaled=pd.DataFrame(df_scaled,columns=selected_col)
        df_s=self.df_restaurant_LA[['negative','neutral','positive','compound']]
        df_s
       
        self.df_LA_scaled=pd.concat([df_s,df_scaled], axis=1)
        self.df_LA_scaled=self.df_LA_scaled[['stars_x','stars_y_userRating','useful','negative','neutral','positive','compound','stars_y_userMean']]
        self.df_LA_scaled.head()
        return self.df_LA_scaled
    
    #Use Sentiment Intensity Analyzer to get sentiment scores.
    # return a result in dictionary form :{neg: _ , neu: _ ,pos: _ ,compound: _ }
    def get_sentiment_scores(self,sentence):
        #clear_output(wait=True)
        return self.analyser.polarity_scores(sentence)
    
    def assemble_user_score(self,sentence):
        bAvg_mean=self.df_LA_scaled['stars_x'].mean()
        uAvg_mean=self.df_LA_scaled['stars_y_userMean'].mean()
        useful_mean=self.df_LA_scaled['useful'].mean()
        
        #user_sentiment_score=get_sentiment_scores(sentence)
        user_sentiment_score=sentence
        #print(bAvg_mean,uAvg_mean, useful_mean, user_sentiment_score)
        return bAvg_mean,uAvg_mean, useful_mean, user_sentiment_score
    #drop stars_y_userRating because we are using Random Forest Classifier to predict the user rating. 

    def assemble_similarity_score(self,score):
        bAvg_mean,uAvg_mean, useful_mean, user_sentiment_score=self.assemble_user_score(score)
        df_userScore=pd.DataFrame(columns=['stars_x','useful','negative','neutral','positive','compound','stars_y_userMean'])
        df_userScore.loc[0]={'stars_x':bAvg_mean,
                             'useful':useful_mean,
                             'negative':user_sentiment_score['neg'],
                             'neutral':user_sentiment_score['neu'],
                             'positive':user_sentiment_score['pos'],
                             'compound':user_sentiment_score['compound'],
                             'stars_y_userMean':uAvg_mean}
        #print (df_userScore)
        return df_userScore
    
    
        
    def get_fullUserScore(self,ass_Sim_scr,predict_stars_y):
        predict_stars_y=pd.DataFrame(predict_stars_y,columns=['predict_stars_y'])
        self.fullUserScore=pd.concat([ass_Sim_scr, predict_stars_y],axis=1)
        return self.fullUserScore
    
    

    def get_similarity(self,x):
        df_y=self.df_LA_scaled[['stars_x','useful','negative','neutral','positive','compound','stars_y_userMean','stars_y_userRating']]

        cosineSimilarity=cosine_similarity(x, df_y, dense_output=True)
        euclideanDistance=euclidean_distances(x,df_y)
        
        cs=pd.DataFrame({'cosineSimilarity':cosineSimilarity.reshape(-1, )})
        ed=pd.DataFrame({'euclideanDistance':euclideanDistance.reshape(-1, )})
        

        self.sentiment_cs_index=cs
        
        self.sentiment_ed_index=ed


    
    def get_randomForest_predict(self,assembledSimilarityScore):
        result=self.loaded_rfc.predict([assembledSimilarityScore])
        scaled_result=result/5
        #print (scaled_result)
        return scaled_result
    
    #starts a series of steps to retrieve restaurant data and prepare it for 
    #assembling into suitable format for fufillment response.
    def view_recommendation(self, final_score):
        final_score_index=final_score.index.values.astype(int)
       # print (final_score_index)
        
        similarity_final_scoreList= final_score.values.tolist()
        
        telegram_data=self.restaurant_info(final_score_index,similarity_final_scoreList)
        #telegram_data=self.restaurant_info([137359,99480])
        reply=fulfillment_showRecommendation(telegram_data)
        return reply
        #for this test case, pass to fulfillment_showRecommendation()
        #actual has to pass back to calling object.
    
    
    #retrive the top 2 recommended restaurants' details.
    def restaurant_info(self,final_score_index,similarity_final_scoreList):
            telegram_data={'Card_1st':[],'Card_2nd':[]}

            #telegram card can only show two cards.

            for i, ind in enumerate(final_score_index):
                address,name,stars_x,attributes,categories=self.get_restaurantDetails(ind)
                rest_attributes=self.restaurant_attributes(attributes,categories)
                if i==0:
                    telegram_data['Card_1st']=[address,name,stars_x,rest_attributes,similarity_final_scoreList[0],ind]
                    #print(telegram_data['Card_1st'])
                else:
                    telegram_data['Card_2nd']=[address,name,stars_x,rest_attributes,similarity_final_scoreList[1],ind]
                    #print(telegram_data['Card_2nd'])
            #print (telegram_data)
            return telegram_data
     
    
    #get the restaurants' 'attributes' column data. 
    #Data within this column can give the restaurant some useful descriptors.
    def restaurant_attributes(self,attributes,categories):
        attributesList=[]
        
        dictStr=ast.literal_eval(attributes)
        dictStr_df=pd.DataFrame([dictStr])
        #print (dictStr)
        ambDictStr=dictStr_df.Ambience
        #print (ambDictStr)
        ads=ast.literal_eval(ambDictStr[0])
        #print(ads)
        for k,v in ads.items():
            if v==True:
                attributesList.append(k)
            else:
                None
        #print (attributesList)
        #print (dictStr_df.columns)
        
        #gFM=dictStr_df.GoodForMeal
        try:
            gFM=dictStr_df.GoodForMeal
            g_FM=ast.literal_eval(gFM[0])
            for k,v in g_FM.items():
                    if v==True:
                        attributesList.append(k)
                    else:
                        None
        except:
            None
        
        try:
            bPk=dictStr_df.BusinessParking
            BPK=ast.literal_eval(bPk[0])
            for k,v in BPK.items():
                    if v ==True:
                        k='Business Parking :'+str(k)
                        attributesList.append(k)
                    else:
                        None    
        except:
            None
            attributesList.append(categories)
        return ','.join(attributesList)
    
    


    #mainline logic within the Recommendation_model class.
    #message_list= [sentiment_text, affinity_modifiers, like_text, dislike_text] - for reference during coding
    def query_recommendation_model(self,message_list):
        self.get_df_LA_scaled()
        message = ','.join(message_list[2:])
        
        mood_today=self.get_sentiment_scores(message_list[0])
        dissimilarity_modifier=self.sentiment_text_tester(mood_today) #to modify dissimilarity strictness level.
        
        score=self.get_sentiment_scores(message)
        #print (score)
        #print (assemble_similarity_score(score))
        ass_Sim_scr=self.assemble_similarity_score(score)
        ass_Sim_scr_ravel=ass_Sim_scr.values.ravel()
        predict_stars_y=self.get_randomForest_predict(ass_Sim_scr_ravel)
        fullUserScore=self.get_fullUserScore(ass_Sim_scr,predict_stars_y)
        print('Full User Score (scaled): \n',fullUserScore)
        self.get_similarity(fullUserScore)
        
        #semantic similarity score 
        #like_message, dislike_message= self.get_message_sentiment(message)
        
        like1,dislike1= self.get_message_sentiment(message_list[2])
        like2,dislike2= self.get_message_sentiment(message_list[3])
        
        like_message=','.join([like1,like2])
        dislike_message=','.join([dislike1,dislike2])
        
        ##query_recommendation_model needs parameters like_message and dislike_message
        similar_taste = self.get_semantic_similarity_scores(like_message)
        dissimilar_taste = self.get_dissimilarity_scores(dislike_message)
        
        print ('\nSemantic Similarity Score :\n',similar_taste.sort_values(by=['cosine_similarity_semantic'],ascending=False).head())
        print ('\nDissimilarity Score :\n',dissimilar_taste.sort_values(by=['dissimilarity'],ascending=False).head())


        #design decision to set cosine similarity score greater than 0.1 as dissimilar.

        #dissimilar_taste = dissimilar_taste.query('dissimilarity > .1')
        #similar_taste = similar_taste.drop(dissimilar_taste.index)
        final_score=self.ensemble_similarity(message_list,
                                             similar_taste,
                                             dissimilar_taste,
                                             dissimilarity_modifier)
        
        
        #print ('Top 10 Ensemble Similarity Scores :',final_score[0:10])
        final_score=self.check_sameRestaurant(final_score[0:6])
        print ('Index, Similarity score, name:\n',final_score)
        
        reply=self.view_recommendation(final_score[0:2]) #only top 2 results are recommended.
        
        return reply
        
    
    
        #clean text
    def stem_words(self,text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text

    def make_lower_case(self,text):
        return text.lower()

    def remove_stop_words(self,text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text

    def remove_punctuation(self,text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        text = " ".join(text)
        return text
    
    def clean_user_message(self, message):
        message=self.make_lower_case(message)
        message =self.remove_stop_words(message)
        message=self.remove_punctuation(message)
        message=self.stem_words(message)
        return message
    
    #infer vector of the user message using doc2vec model.
    def get_message_doc2vec_embedding_vector(self,message):
        message_array = self.doc2vecModel.infer_vector(doc_words=message.split(" "), epochs=200)
        message_array = message_array.reshape(1, -1)
        return message_array
    
    
    def get_semantic_similarity_scores(self, message):
        message = self.clean_user_message(message)
        
        semantic_message_array = self.get_message_doc2vec_embedding_vector(message)
        
        semantic_similarity = self.get_similarity_scores(semantic_message_array, self.doc2vec_featureMatrix)
        
        #don't sort.
        #semantic_similarity.sort_values(by="cosine_similarity", ascending=False, inplace=True)
        
        #print ('Semantic Similarity Score',semantic_similarity)
        return semantic_similarity
    
    
    #calculate the similiary score using cosine similarity compared against the 
    #doc2vec_trained featureMatix. This was trained previously using exisitng text review.
    def get_similarity_scores(self,message_array, embeddings):
        cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=message_array,
                                                           dense_output=True))
        cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        cosine_sim_matrix.columns = ["cosine_similarity_semantic"]
        
        return cosine_sim_matrix
    
    #similiar thinkin as get_similarity_scores except it's applied to get dissimilarity. 
    def get_dissimilarity_scores(self, message):
        message = self.clean_user_message(message)
        semantic_message_array = self.get_message_doc2vec_embedding_vector(message)

        dissimilarity = self.get_similarity_scores(semantic_message_array, self.doc2vec_featureMatrix)
        dissimilarity.columns = ["dissimilarity"]
        
        #dissimilarity.sort_values(by="dissimilarity", ascending=False, inplace=True)
        return dissimilarity
    
    
    #cleans up and user text input and tease out the like and dislike message from sentiment score. 
    def get_message_sentiment(self, message):
        sentences=re.split(r'\s+',re.sub(r'\W+'," ",message))
        sentences = [x for x in sentences if x != ""]
        like_message = ""
        dislike_message = ""
        for s in sentences:
            sentiment_scores = self.get_sentiment_scores(s)
            if sentiment_scores['neg'] > 0:
                dislike_message = dislike_message + s
            else:
                like_message = like_message + s
        return like_message, dislike_message
    
    
    #this function adjust the dissimilarity modifier which restrict datasets used to compute similarity between users. 
    def sentiment_text_tester(self,mood_today):
    
        if float(mood_today['compound'])>=0.05: #when mood is good, we relaxed the recommendation so user can explore.
            dissimilarity_modifier=0.007
        elif float(mood_today['compound'])<=-0.05: #when mood is bad, we are stricter with negative recommendation.
            dissimilarity_modifier=0.003
        else:
            dissimilarity_modifier=0.005  #default value
        return dissimilarity_modifier
    
    
    #calculate the ensemble similiarity score with the modifiers.
    # 2 modifiers are used: dissimilarity modifer, and
    #message_list[1], which is an affinity modifier in dictionary forma 
    #which modifies the weightage between 
    #composite ratings biasedness or semantic biasedness. 
    def ensemble_similarity(self,message_list,similiar_taste, dissimilar_taste,dissimilarity_modifier):
        
        
        #merge cosine similarity, euclidean distance, similar_taste and dissimilar_taste intoe one dataframe.
        df_ensemble_similarity_score=pd.merge(self.sentiment_cs_index,self.sentiment_ed_index, right_index=True, left_index=True)
        df_ensemble_similarity_score=pd.merge(df_ensemble_similarity_score,similiar_taste, right_index=True, left_index=True)
        df_ensemble_similarity_score=pd.merge(df_ensemble_similarity_score,dissimilar_taste, right_index=True, left_index=True)
        
        dissimilarity_modifier= ' > '.join(['dissimilarity',str(dissimilarity_modifier)])
        dissimilar_taste_ind = df_ensemble_similarity_score.query(dissimilarity_modifier)

        df_ensemble_similarity_score = df_ensemble_similarity_score.drop(dissimilar_taste_ind.index)
        
        a=message_list[1]['cs_ed_modifier']
        b=message_list[1]['semantic_modifier']
        
        #the following print statements are to visualise the modifiers. 
        print ('\ndissimilarity_modifier :',dissimilarity_modifier)
        print ('\ncs_ed_modifier :',a)
        print('\nsemantic_modifier :',b)
        
        #we are looking at relative similarity. So similarity score just implies relative sameness or difference
        #it does not imply any relationship between the numbers. ie. 0.8 is not 2x more similar than 0.4. 
        #for similarity score which were previously negative, minmax_scale will tranform to positive or zero.
        #this does not affect ranking performance because the value will rank lower.    
        df_ensemble_similarity_score['cosineSimilarity']=minmax_scale(df_ensemble_similarity_score['cosineSimilarity'])
        df_ensemble_similarity_score['euclideanDistance']=minmax_scale(df_ensemble_similarity_score['euclideanDistance'])
        df_ensemble_similarity_score['cosine_similarity_semantic']=minmax_scale(df_ensemble_similarity_score['cosine_similarity_semantic'])

        #print("df_ensemble_similarity_score['cosineSimilarity'] :\n:",df_ensemble_similarity_score['cosineSimilarity'].max(), df_ensemble_similarity_score['cosineSimilarity'].min())
        #print ("df_ensemble_similarity_score['euclideanDistance'] :\n",df_ensemble_similarity_score['euclideanDistance'].max(),df_ensemble_similarity_score['euclideanDistance'].min())
        #print ("df_ensemble_similarity_score['cosine_similarity_semantic'] :\n",df_ensemble_similarity_score['cosine_similarity_semantic'].max(),df_ensemble_similarity_score['cosine_similarity_semantic'].min())
        
        df_ensemble_similarity_score=((df_ensemble_similarity_score['cosineSimilarity']+df_ensemble_similarity_score['euclideanDistance'])*a/2+df_ensemble_similarity_score['cosine_similarity_semantic']*b)/2
        df_ensemble_similarity_score=pd.DataFrame({'Complete_Similarity_Score': df_ensemble_similarity_score})
        
        
        df_ensemble_similarity_score.sort_values(by="Complete_Similarity_Score", ascending=False, inplace=True)
        print ('\ndf_ensemble_similarity_score :\n',df_ensemble_similarity_score[0:5])
        return df_ensemble_similarity_score
    
    
    def create_connection(self,db_file):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)


    def get_restaurantDetails(self,final_score_index):
        #final_score_index parameter must be passed in as just one integer value.
        sql='''
            SELECT address, name, stars_x, attributes, categories  
            FROM restaurant_details
            WHERE restaurant_details.id=='''+str(final_score_index)
            


        database = "/home/kenaimachine/soulfood/restaurants.db"
        conn = self.create_connection(database)
        with conn:
            cur = conn.cursor()
            cur.execute(sql)
            row = cur.fetchall()
            #print(row)
            
        address=row[0][0]
        name=row[0][1]
        stars_x=row[0][2]
        attributes=row[0][3]
        categories=row[0][4]
            
            #print (address)
            #print (name)
            #print (stars_x)
            #print(attributes)
            #print (categories)


        return address,name,stars_x,attributes,categories
    
    def check_sameRestaurant(self, final_score):
        final_score_index=final_score.index.values
        name_list=[]
        for i in final_score_index:
            sql="SELECT name FROM restaurant_details WHERE restaurant_details.id=="+str(i)
            database = "/home/kenaimachine/soulfood/restaurants.db"
            conn = self.create_connection(database)
            with conn:
                cur = conn.cursor()
                cur.execute(sql)
                row = cur.fetchall()
                name=row[0]
                name_list.append(name)
        
        final_score['name']=name_list
        final_score=final_score.drop_duplicates(subset='name',keep='first')
        #final_score df with col index,similarityscore,name
        return final_score


# #get the photo link -method 2
# #for example name: telegram_data['Card_1st'][1]
# #http://kenaimachine.pythonanywhere.com/static/-1nmjnrNUcmbjYVmlqpkYQ.jpg
# 
# def get_photos(row_number):
#     df_restaurant_LA=df_=pd.read_csv('/home/kenaimachine/soulfood/df_restaurant_LA_buildfinal.csv',low_memory=False)
#     data_df=pd.read_csv('/home/kenaimachine/soulfood/photos.csv',low_memory=False)
# 
#     photo_id=business_picker(row_number, df_restaurant_LA,data_df)
#     return 'http://kenaimachine.pythonanywhere.com/static/'+photo_id

# def business_picker (row_number,df_restaurant_LA, data_df):
#     #print (df_restaurant_LA.index==row_number)
#     a=df_restaurant_LA.loc[df_restaurant_LA.index==row_number, 'business_id']
#     
#     return photo_picker(a[row_number],data_df)
#     

# def photo_picker(id, data_df):
#     a=data_df[data_df.loc[:,'business_id']==id]
#     b=a.loc[a.loc[:,'label']=='outside','photo_id']
#     c=b.reset_index()
#     return c.loc[0]

# In[122]:


#compile and format the required fulfillment messages for dialogflow. 
def fulfillment_showRecommendation(telegram_data):
    #telegram_data={'Card_1st':[address,name,stars_x,rest_attributes,similarity_final_scoreList[0],rowindex_1],'Card_2nd':[address,name,stars_x,rest_attributes,similarity_final_scoreList[1],rowindex_2]}
    #print (type(telegram_data['Card_1st'][0]), type(telegram_data['Card_1st'][3]), type(str(telegram_data['Card_1st'][2])))
    #print ('fulfillment_showRecommendation :',telegram_data)
    
    reply={}
    reply["fulfillmentText"]=" "
    reply["fulfillmentMessages"]= [
             {
        "text": {
          "text": [
            "Thanks for waiting. Here's my recommendations:"
          ]
        },
        "platform": "TELEGRAM"
      },
      {
        "card": {
          "title": telegram_data['Card_1st'][1],
          "subtitle": ' '.join([telegram_data['Card_1st'][0], telegram_data['Card_1st'][3], ' Average Rating :', str(telegram_data['Card_1st'][2]), ' Similarity :', str(telegram_data['Card_1st'][4][0])]),
          "imageUri": get_random_pictures(),#get_photos(telegram_data['Card_1st'][5]),
          "buttons":[
                      {
                        "text": 'https://www.yelp.com/search?find_desc='+ telegram_data['Card_1st'][1]+'&find_loc=Las%20Vegas%2C%20NV%2C%20United%20States&ns=1&cflt=restaurants'
                      }
                     
                      ]
            
        },
        "platform": "TELEGRAM"
      },
      {
        "card": {
          "title": telegram_data['Card_2nd'][1],
          "subtitle": ' '.join([telegram_data['Card_2nd'][0], telegram_data['Card_2nd'][3], ' Average Rating :', str(telegram_data['Card_2nd'][2]), 'Similarity :',str(telegram_data['Card_2nd'][4][0])]),
          "imageUri": get_random_pictures(),
          "buttons":[
                      {
                        "text": 'https://www.yelp.com/search?find_desc='+ telegram_data['Card_2nd'][1]+'&find_loc=Las%20Vegas%2C%20NV%2C%20United%20States&ns=1&cflt=restaurants'
                      }
                     
                      ]
            
            
        },
        "platform": "TELEGRAM"
      },
      {
        "text": {
          "text": [
            "I hoped you like the two recommendations. Have a great day! To restart, type /restart."
          ]
        },
        "platform": "TELEGRAM"
      }
        
        ]
    
    #print (reply)
    return reply
    ##return jsonify(reply)    

    #require, title, subtitle, imageUrl, button text


# #compile and format the required fulfillment messages for dialogflow. 
# def fulfillment_showRecommendation(telegram_data):
#     #telegram_data={'Card_1st':[address,name,stars_x,rest_attributes],'Card_2nd':[address,name,stars_x,rest_attributes]}
#     #print (type(telegram_data['Card_1st'][0]), type(telegram_data['Card_1st'][3]), type(str(telegram_data['Card_1st'][2])))
#     #print ('fulfillment_showRecommendation :',telegram_data)
#     
#     reply={}
#     reply["fulfillmentText"]=" "
#     reply["fulfillmentMessages"]= [
#             {
#                 "text": {
#                   "text": [
#                     "Thanks for waiting. After careful analysis, I am giving you top two recommendations. Please see below:"
#                   ]
#                 },
#                 "platform": "TELEGRAM"
#               },
#             { 
#                 
#               "card" : {
#                   "title":telegram_data['Card_1st'][1],
#                   "subtitle":' '.join([telegram_data['Card_1st'][0], telegram_data['Card_1st'][3], ' Average Rating :', str(telegram_data['Card_1st'][2])]),
#                   "imageUrl": get_random_pictures(),
#                   "buttons":[
#                       {
#                         "text": 'https://www.yelp.com/search?find_desc='+ telegram_data['Card_1st'][1]+'&find_loc=Las%20Vegas%2C%20NV%2C%20United%20States&ns=1&cflt=restaurants'
#                       }
#                      
#                       ]
#                   },
#                 "platform":"TELEGRAM"
#             },
#             {
#             "card" : {
#                   "title":telegram_data['Card_2nd'][1],
#                   "subtitle":' '.join([telegram_data['Card_2nd'][0], telegram_data['Card_2nd'][3], ' Average Rating :', str(telegram_data['Card_2nd'][2])]),
#                   "imageUrl": get_random_pictures(),
#                   "buttons":[
#                       {
#                         "text": 'https://www.yelp.com/search?find_desc='+ telegram_data['Card_2nd'][1]+'&find_loc=Las%20Vegas%2C%20NV%2C%20United%20States&ns=1&cflt=restaurants'
#                       }
#                      
#                       ]
#                   },
#             "platform":"TELEGRAM"
#             }
#         
#         ]
#     
#     #print (reply)
#     return reply
#     ##return jsonify(reply)    
# 
#     #require, title, subtitle, imageUrl, button text

# In[49]:


if __name__ == '__main__':
    app.run()


# #tester
# sentiment_text_1='hello u'
# sentiment_text_2='gd bye'
# sentiment_text_3='abc, def'
# sentiment_text_4=[sentiment_text_1,sentiment_text_2,sentiment_text_3]
# 
# sentiment_text=','.join([sentiment_text_1,sentiment_text_2,sentiment_text_3])
# print (sentiment_text, type(sentiment_text))
# message = ','.join(sentiment_text_4[1:])
# print (message)
# 

# #tester
# import random, os
# from PIL import Image
# photo=random.choice([x for x in os.listdir("/users/KennethMacBookPro/Downloads/Capstone Project/images")
#                if os.path.isfile(os.path.join("/users/KennethMacBookPro/Downloads/Capstone Project/images", x))])
# #print (photo)
# 
# Image.open("/users/KennethMacBookPro/Downloads/Capstone Project/images/"+photo)

# #For testing purposes. 
# data={
#   "responseId": "1d0decfd-76ec-4c3c-8a02-4fd0442a265c-b55300fa",
#   "queryResult": {
#     "queryText": "ok",
#     "action": "get_tasks",
#     "parameters": {},
#     "allRequiredParamsPresent": True,
#     "fulfillmentMessages": [
#       {
#         "text": {
#           "text": [
#             ""
#           ]
#         }
#       }
#     ],
#     "outputContexts": [
#       {
#         "name": "projects/restaurant-recommender-duusew/agent/sessions/12418c30-10f7-9e44-9526-d63f2cbaf759/contexts/ctx_like_text",
#         "lifespanCount": 5,
#         "parameters": {
#           "like_text.original": "like pizza",
#           "dislike_text.original": "dislike burgers",
#           "like_text": "like pizza",
#           "dislike_text": "dislike burgers"
#         }
#       },
#       {
#         "name": "projects/restaurant-recommender-duusew/agent/sessions/12418c30-10f7-9e44-9526-d63f2cbaf759/contexts/ctx_dislike_text",
#         "lifespanCount": 5,
#         "parameters": {
#           "dislike_text.original": "dislike burgers",
#           "dislike_text": "dislike burgers"
#         }
#       },
#       {
#         "name": "projects/restaurant-recommender-duusew/agent/sessions/12418c30-10f7-9e44-9526-d63f2cbaf759/contexts/ctx_affinity_test",
#         "lifespanCount": 5,
#         "parameters": {
#           "dislike_text.original": "dislike burgers",
#           "like_text": "like pizza",
#           "affinity_test_1.original": "1",
#           "like_text.original": "like pizza",
#           "affinity_test_1": "1",
#           "affinity_test_2.original": "5",
#           "dislike_text": "dislike burgers",
#           "affinity_test_2": "5"
#         }
#       },
#       {
#         "name": "projects/restaurant-recommender-duusew/agent/sessions/12418c30-10f7-9e44-9526-d63f2cbaf759/contexts/ctx_sentiment_text",
#         "lifespanCount": 5,
#         "parameters": {
#           "sentiment_text_3": "cheerful",
#           "affinity_test_1": "1",
#           "affinity_test_2": "5",
#           "like_text": "like pizza",
#           "sentiment_text_1.original": "happy",
#           "affinity_test_1.original": "1",
#           "sentiment_text_1": "cheerful",
#           "like_text.original": "like pizza",
#           "sentiment_text_3.original": "happy",
#           "affinity_test_2.original": "5",
#           "dislike_text": "dislike burgers",
#           "sentiment_text_2": "cheerful",
#           "sentiment_text_2.original": "happy",
#           "dislike_text.original": "dislike burgers"
#         }
#       },
#       {
#         "name": "projects/restaurant-recommender-duusew/agent/sessions/12418c30-10f7-9e44-9526-d63f2cbaf759/contexts/asktoproceed",
#         "lifespanCount": 5,
#         "parameters": {
#           "dislike_text.original": "dislike burgers",
#           "dislike_text": "dislike burgers"
#         }
#       }
#     ],
#     "intent": {
#       "name": "projects/restaurant-recommender-duusew/agent/intents/807ddd54-370f-49a3-b1f4-6ce60f8bde5f",
#       "displayName": "MakeRecommendations"
#     },
#     "intentDetectionConfidence": 1,
#     "diagnosticInfo": {
#       "webhook_latency_ms": 607
#     },
#     "languageCode": "en"
#   },
#   "webhookStatus": {
#     "code": 13,
#     "message": "Webhook call failed. Error: 500 Internal Server Error."
#   }
# }
# 
# 

# #for testing only
# get_tasks(data)

# In[ ]:





# def on_button_clicked(b):
#     clear_output()
#     print ("Describe the restaurant experience you are looking for. You can be as detailed as you like! ")
#     display(text)
#     display(button)
#      
# 
# def handle_submit(sender):
#     print ("Got it! Hold tight while I find your recommendations!")
#     message = sender.value
#     rm.query_recommendation_model(message)
# 
# 
# 
# 

# rm=Recommender_Model()
# print ("Describe the restaurant experience you are looking for. You can be as detailed as you like!\n\
# Type In The Box Provided And Press Enter To Submit:")
# text = widgets.Text()
# display(text)
# button = widgets.Button(description="Restart!")
# display(button)
# 
# 
# text.on_submit(handle_submit)
# button.on_click(on_button_clicked)
# 
# 

# In[ ]:




