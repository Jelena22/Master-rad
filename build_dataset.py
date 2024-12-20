import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def create_dataset(data_frame1, data_frame2, data_frame3):
    result_df = pd.concat([data_frame1, data_frame2], axis=0, ignore_index=True)

    df3 = result_df.drop(['Category', 'Nominee', 'Release_date',
                          #'Nom_DGA','Nom_BAFTA', 'Win_DGA', 'Win_BAFTA', 'Nom_GoldenGlobe_bestcomedy',
                          #'Nom_GoldenGlobe_bestdrama', 'Win_GoldenGlobe_bestcomedy','Win_GoldenGlobe_bestdrama',
                          'MPAA_rating','MPAA_G', 'MPAA_PG', 'MPAA_PG-13', 'MPAA_R','MPAA_NC-17',
                          #'Nowin_Criticschoice','Win_Criticschoice', 'Nonom_Criticschoice', 'Nom_Criticschoice',
                          #'Nowin_SAG_bestcast','Win_SAG_bestcast', 'Nonom_SAG_bestcast','Nom_SAG_bestcast',
                          #'Nowin_PGA', 'Win_PGA', 'Nonom_PGA', 'Nom_PGA'
                          ], axis=1)

    # print(df3)

    data_frame3.rename(columns={'original_title': 'Film'}, inplace=True)
    # pd.set_option('display.max_columns', None)   #da se prikazu sve kolone

    # print(dataFrame3)

    selected_columns = ['Film', 'duration']
    new_dataFrame = data_frame3[selected_columns]
    # print(new_dataFrame)

    merged_df = pd.merge(df3, new_dataFrame, on='Film', how='left')

    pd.set_option('display.max_columns', None)
    print(merged_df)
    return merged_df

def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=100)

# def tf_idf_transformer(train_texts, test_texts):
#     # Preuzimanje zaustavnih reči iz nltk
#     stop_words = set(stopwords.words('english'))
#
#     # Tokenizacija i uklanjanje zaustavnih reči
#     train_texts_tokenized = [word_tokenize(text.lower()) for text in train_texts.astype(str)]
#     train_texts_filtered = [[word for word in text if word.isalnum() and word not in stop_words] for text in
#                             train_texts_tokenized]
#
#     test_texts_tokenized = [word_tokenize(text.lower()) for text in test_texts.astype(str)]
#     test_texts_filtered = [[word for word in text if word.isalnum() and word not in stop_words] for text in
#                            test_texts_tokenized]
#
#     # Inicijalizacija i primena TfidfVectorizer-a na treninški skup
#     vectorizer = TfidfVectorizer()
#     X_train_tfidf = vectorizer.fit_transform([' '.join(text) for text in train_texts_filtered])
#
#     # Transformacija testnog skupa pomoću fitovanog vektorizatora
#     X_test_tfidf = vectorizer.transform([' '.join(text) for text in test_texts_filtered])
#
#     return X_train_tfidf, X_test_tfidf, vectorizer

def create_dataset_actors(data_frame4, data_frame5):
    result_df = pd.concat([data_frame4, data_frame5], axis=0, ignore_index=True)

    df3 = result_df.drop(['Release_date',
                          # 'Nom_DGA','Nom_BAFTA', 'Win_DGA', 'Win_BAFTA', 'Nom_GoldenGlobe_bestcomedy',
                          # 'Nom_GoldenGlobe_bestdrama', 'Win_GoldenGlobe_bestcomedy','Win_GoldenGlobe_bestdrama',
                          'MPAA_rating', 'MPAA_G', 'MPAA_PG', 'MPAA_PG-13', 'MPAA_R', 'MPAA_NC-17',
                          # 'Nowin_Criticschoice','Win_Criticschoice', 'Nonom_Criticschoice', 'Nom_Criticschoice',
                          # 'Nowin_SAG_bestcast','Win_SAG_bestcast', 'Nonom_SAG_bestcast','Nom_SAG_bestcast',
                          # 'Nowin_PGA', 'Win_PGA', 'Nonom_PGA', 'Nom_PGA'
                          ], axis=1)

    print(df3)

    return df3