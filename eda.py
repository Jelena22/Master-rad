import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def fill_missing_values(df, duration_mean=None):
    # Transformacija kolone 'duration' u numerički format i popunjavanje nedostajućih vrednosti
    df['duration'] = df['duration'].str.replace(' min.', '').astype(float)
    if duration_mean is None:  # Ako nije zadato, računamo srednju vrednost trajanja samo za trening skup
        duration_mean = df['duration'].mean()
    df['duration'].fillna(duration_mean, inplace=True)

    # Stvaranje novih kolona sa transformisanim vrednostima
    df['Sqrt_Duration'] = np.sqrt(df['duration'])
    df['Log_Rating_rtaudience'] = np.log(df['Rating_rtaudience'])
    df['Log_Rating_rtcritic'] = np.log(df['Rating_rtcritic'])
    df['Log_Rating_IMDB'] = np.log(df['Rating_IMDB'])

    # Brisanje originalnih kolona koje su transformisane
    df = df.drop(['duration', 'Rating_rtaudience', 'Rating_rtcritic', 'Rating_IMDB'], axis=1)
    return df, duration_mean

def fill_missing_values_actors(df, imputer=None):
    # Primena transformacija i kreiranje novih kolona
    df['Sqrt_Rating_rtaudience'] = np.sqrt(df['Rating_rtaudience'])
    df['Log_Rating_rtcritic'] = np.log(df['Rating_rtcritic'])
    df['Log_Rating_IMDB'] = np.log(df['Rating_IMDB'])
    df = df.drop(['Rating_rtaudience', 'Rating_rtcritic', 'Rating_IMDB'], axis=1)

    # Kreiranje ili primena imputera
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        df[['Nom_GoldenGlobe_comedy-leadacting', 'Nom_GoldenGlobe_drama-leadacting',
            'Win_GoldenGlobe_comedy-leadacting', 'Win_GoldenGlobe_drama-leadacting',
            'Birthyear', 'Age']] = imputer.fit_transform(df[['Nom_GoldenGlobe_comedy-leadacting',
                                                             'Nom_GoldenGlobe_drama-leadacting',
                                                             'Win_GoldenGlobe_comedy-leadacting',
                                                             'Win_GoldenGlobe_drama-leadacting',
                                                             'Birthyear', 'Age']])
    else:
        df[['Nom_GoldenGlobe_comedy-leadacting', 'Nom_GoldenGlobe_drama-leadacting',
            'Win_GoldenGlobe_comedy-leadacting', 'Win_GoldenGlobe_drama-leadacting',
            'Birthyear', 'Age']] = imputer.transform(df[['Nom_GoldenGlobe_comedy-leadacting',
                                                         'Nom_GoldenGlobe_drama-leadacting',
                                                         'Win_GoldenGlobe_comedy-leadacting',
                                                         'Win_GoldenGlobe_drama-leadacting',
                                                         'Birthyear', 'Age']])
    return df, imputer


# Funkcija za one-hot enkodiranje
def encode_categorical_features(df, encoder=None):
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_columns = encoder.fit_transform(df[['Category', 'Nominee']])
    else:
        encoded_columns = encoder.transform(df[['Category', 'Nominee']])

    # Dodavanje enkodiranih kolona u DataFrame i uklanjanje originalnih kolona
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['Category', 'Nominee']))
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(['Category', 'Nominee'], axis=1)

    return df, encoder

def tf_idf_transformer(train_texts, test_texts):
    # Preuzimanje zaustavnih reči iz nltk
    stop_words = set(stopwords.words('english'))

    # Tokenizacija i uklanjanje zaustavnih reči
    train_texts_tokenized = [word_tokenize(text.lower()) for text in train_texts.astype(str)]
    train_texts_filtered = [[word for word in text if word.isalnum() and word not in stop_words] for text in
                            train_texts_tokenized]

    test_texts_tokenized = [word_tokenize(text.lower()) for text in test_texts.astype(str)]
    test_texts_filtered = [[word for word in text if word.isalnum() and word not in stop_words] for text in
                           test_texts_tokenized]

    # Inicijalizacija i primena TfidfVectorizer-a na treninški skup
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform([' '.join(text) for text in train_texts_filtered])

    # Transformacija testnog skupa pomoću fitovanog vektorizatora
    X_test_tfidf = vectorizer.transform([' '.join(text) for text in test_texts_filtered])

    return X_train_tfidf, X_test_tfidf, vectorizer

def visualization(merged_df):

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=merged_df, x='Rating_IMDB', hue='Winner', bins=30, kde=True)
    plt.title('Histogram IMDB ocena u odnosu na osvajanje Oskara')
    plt.xlabel('IMDB ocene')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.histplot(data=merged_df, x='Rating_rtaudience', hue='Winner', bins=30, kde=True)
    plt.title('Histogram ocena publike Rotten Tomatoes u odnosu na osvajanje Oskara')
    plt.xlabel('Rating_rtaudience')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.histplot(data=merged_df, x='Rating_rtcritic', hue='Winner', bins=30, kde=True)
    plt.title('Histogram ocena Rotten Tomatoes critic u odnosu na osvajanje Oskara')
    plt.xlabel('Rating_rtcritic')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x='Oscarstat_totalnoms', hue='Winner', bins=30, kde=True)
    plt.title('Broj nominacija za Oskara u odnosu na osvajanje Oskara')
    plt.xlabel('Broj nominacija za Oskara')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x='duration', hue='Winner', bins=30, kde=True)
    plt.title('Trajanje filma u odnosu na osvajanje Oskara')
    plt.xlabel('trajanje filma')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Nom_Oscar_bestdirector', hue='Winner', data=merged_df)
    plt.title('Nominacija za Oskara za najboljeg režisera u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Nominacija za Oskara za najboljeg režisera')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(x='Genre_drama', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u dramu u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Žanr drama')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(x='Genre_comedy', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u komediju u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Žanr komedija')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(x='Genre_sci-fi', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u sci-fi u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Žanr naučna fantastika')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(x='Genre_biography', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u biografiju u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Žanr biografija')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(x='Nom_GoldenGlobe_bestcomedy', hue='Winner', data=merged_df)
    plt.title('Nominacija Golden Globe za najbolju komediju u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Nominacija Golden Globe za najbolju komediju')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(x='Win_GoldenGlobe_bestcomedy', hue='Winner', data=merged_df)
    plt.title('Osvajanje Golden Globe za najbolju komediju u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Osvajanje Golden Globe za najbolju komediju')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(x='Nom_GoldenGlobe_bestdrama', hue='Winner', data=merged_df)
    plt.title('Nominacija Golden Globe za najbolju dramu u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Nominacija Golden Globe za najbolju dramu')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(x='Win_GoldenGlobe_bestdrama', hue='Winner', data=merged_df)
    plt.title('Osvajanje Golden Globe za najbolju dramu u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Osvajanje Golden Globe za najbolju dramu')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nom_DGA', hue='Winner')
    plt.title('Nominacija DGA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Nominacija DGA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Win_DGA', hue='Winner')
    plt.title('Osvajanje DGA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Osvajanje DGA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nom_BAFTA', hue='Winner')
    plt.title('Nominacija BAFTA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Nominacija BAFTA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_BAFTA', hue='Winner')
    plt.title('Osvajanje BAFTA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Osvajanje BAFTA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nonom_Criticschoice', hue='Winner')
    plt.title('Filmovi koji nisu nominovani Criticschoice u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Filmovi koji nisu nominovani Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Nom_Criticschoice', hue='Winner')
    plt.title('Nominovani filmovi Criticschoice u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Nominovani filmovi Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nowin_Criticschoice', hue='Winner')
    plt.title('Filmovi koji nisu osvojili Criticschoice u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Filmovi koji nisu osvojili Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_Criticschoice', hue='Winner')
    plt.title('Filmovi koji osvojili Criticschoice u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Filmovi koji osvojili Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nonom_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava nije nominovana SAG u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Glumačka postava nije nominovana SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Nom_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava nominovana SAG u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Glumačka postava nominovana SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nowin_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava nije osvojila SAG u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Glumačka postava nije osvojila SAG ')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava osvojila SAG u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Glumačka postava osvojila SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nonom_PGA', hue='Winner')
    plt.title('Filmovi koji nisu nominovani PGA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Filmovi koji nisu nominovani PGA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Nom_PGA', hue='Winner')
    plt.title('Filmovi koji nominovani PGA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Glumačka postava nominovana SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nowin_PGA', hue='Winner')
    plt.title('Filmovi koji nisu osvojili PGA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Filmovi koji nisu osvojili PGA ')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_PGA', hue='Winner')
    plt.title('Filmovi koji osvojili PGA u odnosu na osvajanje Oskara za najbolji film')
    plt.xlabel('Filmovi koji osvojili PGA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Winner', data=merged_df)
    plt.title('Broj nominovanih i pobednika')
    plt.show()

    total_per_quarter = merged_df[['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4']].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(total_per_quarter, labels=total_per_quarter.index, autopct='%1.1f%%', startangle=90)
    plt.title('Broj filmova po četvrtinama godine')
    plt.show()

    plt.figure(figsize=(8, 8))
    quarterly_counts = merged_df.groupby(['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4', 'Winner']).size()
    quarterly_counts = quarterly_counts.reset_index(name='Count')
    winners_counts = quarterly_counts[quarterly_counts['Winner'] == 1]
    total_per_quarter = quarterly_counts.groupby(['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4'])[
        'Count'].sum()
    winners_ratio_per_quarter = winners_counts.set_index(['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4'])[
                                    'Count'] / total_per_quarter
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    plt.pie(winners_ratio_per_quarter, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Odnos pobedničkih filmova u četvrtinama godine')
    plt.show()


    total_per_quarter = merged_df[
        ['Genre_action', 'Genre_biography', 'Genre_crime', 'Genre_comedy', 'Genre_drama', 'Genre_fantasy',
         'Genre_sci-fi',
         'Genre_mystery', 'Genre_music', 'Genre_romance', 'Genre_history', 'Genre_war', 'Genre_thriller',
         'Genre_adventure',
         'Genre_family']].sum()
    plt.figure(figsize=(16, 12))
    plt.pie(total_per_quarter, labels=total_per_quarter.index, autopct='%1.1f%%', startangle=90)
    plt.title('Broj filmova po žanrovima')
    plt.show()

    # outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=merged_df['Rating_IMDB'])
    plt.title('Boxplot IMDB Ocjena')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=merged_df['Rating_rtaudience'])
    plt.title('Boxplot Rating_rtaudience')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=merged_df['Rating_rtcritic'])
    plt.title('Boxplot Rating_rtcritic')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=merged_df['Oscarstat_totalnoms'])
    plt.title('Boxplot Oscarstat_totalnoms')
    plt.show()

def visualization_actors(merged_df):
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=merged_df, x='Rating_IMDB', hue='Winner', bins=30, kde=True)
    plt.title('Histogram IMDB ocena u odnosu na osvajanje Oskara')
    plt.xlabel('IMDB ocene')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.histplot(data=merged_df, x='Rating_rtaudience', hue='Winner', bins=30, kde=True)
    plt.title('Histogram ocena publike Rotten Tomatoes u odnosu na osvajanje Oskara')
    plt.xlabel('Rating_rtaudience')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.histplot(data=merged_df, x='Rating_rtcritic', hue='Winner', bins=30, kde=True)
    plt.title('Histogram ocena Rotten Tomatoes critic u odnosu na osvajanje Oskara')
    plt.xlabel('Rating_rtcritic')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x='Oscarstat_totalnoms', hue='Winner', bins=30, kde=True)
    plt.title('Broj nominacija za Oskara u odnosu na osvajanje Oskara')
    plt.xlabel('Broj nominacija za Oskara')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    #pogledaj mozda bolje nekako drugacije prikazati
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=merged_df, x='Oscarstat_previousnominations_acting', hue='Winner', bins=30, kde=True)
    plt.title('Histogram IMDB ocena u odnosu na osvajanje Oskara')
    plt.xlabel('IMDB ocene')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.histplot(data=merged_df, x='Oscarstat_previouswins_acting', hue='Winner', bins=30, kde=True)
    plt.title('Histogram ocena publike Rotten Tomatoes u odnosu na osvajanje Oskara')
    plt.xlabel('Rating_rtaudience')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    # merged_df['Birthyear'] = merged_df['Birthyear'].astype(int)
    # sns.histplot(data=merged_df, x='Birthyear', hue='Winner', multiple='stack',
    #              bins=range(merged_df['Birthyear'].min(), merged_df['Birthyear'].max() + 1, 5))
    # plt.title('Distribucija godina rođenja glumaca u odnosu na osvajanje Oskara')
    # plt.xlabel('Godina rođenja')
    # plt.ylabel('Broj glumaca')
    # plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvojen'])
    sns.histplot(data=merged_df.dropna(subset=['Birthyear']),
                 x='Birthyear', hue='Winner', multiple='stack',
                 bins=range(int(merged_df['Birthyear'].min()),
                            int(merged_df['Birthyear'].max()) + 1, 5))

    plt.title('Distribucija godina rođenja glumaca u odnosu na osvajanje Oskara')
    plt.xlabel('Godina rođenja')
    plt.ylabel('Broj glumaca')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvojen'])

    plt.subplot(2, 2, 2)
    # merged_df['Age'] = merged_df['Age'].astype(int)
    sns.histplot(merged_df.dropna(subset=['Age']), x='Age', hue='Winner', multiple='stack',
                 bins=range(int(merged_df['Age'].min()), int(merged_df['Age'].max()) + 1, 5))
    plt.title('Distribucija godina rođenja glumaca u odnosu na osvajanje Oskara')
    plt.xlabel('Godina rođenja')
    plt.ylabel('Broj glumaca')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvojen'])
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(x='Genre_drama', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u dramu u odnosu na osvajanje Oskara')
    plt.xlabel('Žanr drama')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(x='Genre_comedy', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u komediju u odnosu na osvajanje Oskara')
    plt.xlabel('Žanr komedija')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(x='Genre_sci-fi', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u sci-fi u odnosu na osvajanje Oskara')
    plt.xlabel('Žanr naučna fantastika')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(x='Genre_biography', hue='Winner', data=merged_df)
    plt.title('Filmovi koji spadaju u biografiju u odnosu na osvajanje Oskara')
    plt.xlabel('Žanr biografija')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    supporting_df = merged_df[merged_df['Category'].isin(['Actor', 'Actress'])]
    sns.countplot(x='Nom_GoldenGlobe_comedy-leadacting', hue='Winner', data=supporting_df)
    plt.title('Nominacija Golden Globe za glavnog glumca u komediji u odnosu na osvajanje Oskara')
    plt.xlabel('Nominacija Golden Globe za glavnog glumca u komediji')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    supporting_df = merged_df[merged_df['Category'].isin(['Actor', 'Actress'])]
    sns.countplot(x='Win_GoldenGlobe_comedy-leadacting', hue='Winner', data=supporting_df)
    plt.title('Osvajanje Golden Globe za glavnog glumca u komediji u odnosu na osvajanje Oskara')
    plt.xlabel('Osvajanje Golden Globe za glavnog glumca u komediji')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    supporting_df = merged_df[merged_df['Category'].isin(['Actor', 'Actress'])]
    sns.countplot(x='Nom_GoldenGlobe_drama-leadacting', hue='Winner', data=supporting_df)
    plt.title('Nominacija Golden Globe za glavnog glumca u drami u odnosu na osvajanje Oskara')
    plt.xlabel('Nominacija Golden Globe za glavnog glumca u drami')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    supporting_df = merged_df[merged_df['Category'].isin(['Actor', 'Actress'])]
    sns.countplot(x='Win_GoldenGlobe_drama-leadacting', hue='Winner', data=supporting_df)
    plt.title('Osvajanje Golden Globe za glavnog glumca u drami u odnosu na osvajanje Oskara')
    plt.xlabel('Osvajanje Golden Globe za glavnog glumca u drami')
    plt.ylabel('Broj filmova')
    plt.legend(title='Winner', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    supporting_df = merged_df[merged_df['Category'].isin(['Supporting Actor', 'Supporting Actress'])]
    sns.countplot(data=supporting_df, x='Nom_GoldenGlobe_supportingacting', hue='Winner')
    plt.title('Nominacija Golden Globe za sporednog glumca u odnosu na osvajanje Oskara')
    plt.xlabel('Nominacija Golden Globe za sporednog glumca')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    # Filtriramo DataFrame da sadrži samo glumce i glumice u sporednim ulogama
    supporting_df = merged_df[merged_df['Category'].isin(['Supporting Actor', 'Supporting Actress'])]
    sns.countplot(data=supporting_df, x='Win_GoldenGlobe_supportingacting', hue='Winner')
    plt.title('Osvajanje Golden Globe za sporedne uloge u odnosu na osvajanje Oskara')
    plt.xlabel('Osvajanje Golden Globe')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nom_BAFTA', hue='Winner')
    plt.title('Nominacija BAFTA u odnosu na osvajanje Oskara')
    plt.xlabel('Nominacija BAFTA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_BAFTA', hue='Winner')
    plt.title('Osvajanje BAFTA u odnosu na osvajanje Oskara')
    plt.xlabel('Osvajanje BAFTA')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Oscarstat_previousnominee_acting', hue='Winner')
    plt.title('Glumac ima prethodnih nominacija za Oskara u odnosu na osvajanje Oskara')
    plt.xlabel('Glumac ima prethodnih nominacija za Oskara')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Oscarstat_previouswinner_acting', hue='Winner')
    plt.title('Glumac ima prethodnih osvajanja Oskara u odnosu na osvajanje Oskara')
    plt.xlabel('Glumac ima prethodnih osvajanja Oskara')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nom_Oscar_bestfilm', hue='Winner')
    plt.title('Nominacija Oskara za najbolji film u odnosu na osvajanje Oskara')
    plt.xlabel('Nominacija Oskara za najbolji film')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nonom_Criticschoice', hue='Winner')
    plt.title('Glumci koji nisu nominovani Criticschoice u odnosu na osvajanje Oskara')
    plt.xlabel('Glumci koji nisu nominovani Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Nom_Criticschoice', hue='Winner')
    plt.title('Nominovani glumci Criticschoice u odnosu na osvajanje Oskara')
    plt.xlabel('Nominovani glumci Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nowin_Criticschoice', hue='Winner')
    plt.title('Glumci koji nisu osvojili Criticschoice u odnosu na osvajanje Oskara')
    plt.xlabel('Glumci koji nisu osvojili Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_Criticschoice', hue='Winner')
    plt.title('Glumci koji su osvojili Criticschoice u odnosu na osvajanje Oskara')
    plt.xlabel('Glumci koji su osvojili Criticschoice')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nonom_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava nije nominovana SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumačka postava nije nominovana SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Nom_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava nominovana SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumačka postava nominovana SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nowin_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava nije osvojila SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumačka postava nije osvojila SAG ')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_SAG_bestcast', hue='Winner')
    plt.title('Glumačka postava osvojila SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumačka postava osvojila SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Nonom_SAG_acting', hue='Winner')
    plt.title('Glumac nije nominovan SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumac nije nominovan SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Nom_SAG_acting', hue='Winner')
    plt.title('Glumac nominovan SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumac nominovana SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 3)
    sns.countplot(data=merged_df, x='Nowin_SAG_acting', hue='Winner')
    plt.title('Glumac nije osvojio SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumac nije osvojio SAG ')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])

    plt.subplot(2, 2, 4)
    sns.countplot(data=merged_df, x='Win_SAG_acting', hue='Winner')
    plt.title('Glumac osvojio SAG u odnosu na osvajanje Oskara')
    plt.xlabel('Glumac osvojilo SAG')
    plt.ylabel('Broj filmova')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvajanje'])
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Winner', data=merged_df)
    plt.title('Broj nominovanih i pobednika')
    plt.show()

    total_per_quarter = merged_df[['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4']].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(total_per_quarter, labels=total_per_quarter.index, autopct='%1.1f%%', startangle=90)
    plt.title('Broj nominovanih glumaca za Oskara po četvrtinama godine')
    plt.show()

    plt.figure(figsize=(8, 8))
    quarterly_counts = merged_df.groupby(['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4', 'Winner']).size()
    quarterly_counts = quarterly_counts.reset_index(name='Count')
    winners_counts = quarterly_counts[quarterly_counts['Winner'] == 1]
    total_per_quarter = quarterly_counts.groupby(['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4'])[
        'Count'].sum()
    winners_ratio_per_quarter = winners_counts.set_index(['Release_Q1', 'Release_Q2', 'Release_Q3', 'Release_Q4'])[
                                    'Count'] / total_per_quarter
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    plt.pie(winners_ratio_per_quarter, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Odnos glumaca koji su osvojili Oskara po četvrtinama godine')
    plt.show()

    #print(merged_df.columns)

    # total_per_quarter = merged_df[
    #     ['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]',
    #                                       'Age_[65-75] ', 'Age_[75+] ']].sum()
    # plt.figure(figsize=(16, 12))
    # plt.pie(total_per_quarter, labels=total_per_quarter.index, autopct='%1.1f%%', startangle=90)
    # plt.title('Nominovani glumci za Oskara po godištu')
    # plt.show()
    #
    # plt.figure(figsize=(16, 12))
    # quarterly_counts = merged_df.groupby(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]',
    #                                       'Age_[65-75] ', 'Age_[75+] ', 'Winner']).size()
    # quarterly_counts = quarterly_counts.reset_index(name='Count')
    # winners_counts = quarterly_counts[quarterly_counts['Winner'] == 1]
    # total_per_quarter = quarterly_counts.groupby(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]',
    #                                               'Age_[55-65]', 'Age_[65-75] ', 'Age_[75+] '])[
    #     'Count'].sum()
    # winners_ratio_per_quarter = winners_counts.set_index(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]',
    #                                                       'Age_[55-65]', 'Age_[65-75] ', 'Age_[75+] '])[
    #                                 'Count'] / total_per_quarter
    # labels = ['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]', 'Age_[65-75]', 'Age_[75+]']
    # plt.pie(winners_ratio_per_quarter, labels=labels, autopct='%1.1f%%', startangle=90)
    # plt.title('Odnos glumaca koji su osvojili Oskara po godištu')
    # plt.show()

    # Filtrirajte podatke da uključite samo žene
    filtered_df = merged_df[merged_df['Female'] == 1]
    plt.figure(figsize=(16, 12))
    quarterly_counts = filtered_df.groupby(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]',
                                            'Age_[65-75] ', 'Age_[75+] ', 'Winner']).size()
    quarterly_counts = quarterly_counts.reset_index(name='Count')
    winners_counts = quarterly_counts[quarterly_counts['Winner'] == 1]
    total_per_quarter = quarterly_counts.groupby(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]',
                                                  'Age_[55-65]', 'Age_[65-75] ', 'Age_[75+] '])['Count'].sum()
    winners_ratio_per_quarter = winners_counts.set_index(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]',
                                                          'Age_[55-65]', 'Age_[65-75] ', 'Age_[75+] '])[
                                    'Count'] / total_per_quarter

    # Definišite label-e za grafikon
    labels = ['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]', 'Age_[65-75]', 'Age_[75+]']

    # Prikazivanje pie grafika
    plt.pie(winners_ratio_per_quarter, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Odnos glumica koje su osvojile Oskara po godištu')
    plt.show()

    # Filtrirajte podatke da uključite samo žene
    filtered_df = merged_df[merged_df['Female'] == 0]
    plt.figure(figsize=(16, 12))
    quarterly_counts = filtered_df.groupby(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]',
                                            'Age_[65-75] ', 'Age_[75+] ', 'Winner']).size()
    quarterly_counts = quarterly_counts.reset_index(name='Count')
    winners_counts = quarterly_counts[quarterly_counts['Winner'] == 1]
    total_per_quarter = quarterly_counts.groupby(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]',
                                                  'Age_[55-65]', 'Age_[65-75] ', 'Age_[75+] '])['Count'].sum()
    winners_ratio_per_quarter = winners_counts.set_index(['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]',
                                                          'Age_[55-65]', 'Age_[65-75] ', 'Age_[75+] '])[
                                    'Count'] / total_per_quarter

    # Definišite label-e za grafikon
    labels = ['Age_[0-25]', 'Age_[25-35]', 'Age_[35-45]', 'Age_[45-55]', 'Age_[55-65]', 'Age_[65-75]', 'Age_[75+]']

    # Prikazivanje pie grafika
    plt.pie(winners_ratio_per_quarter, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Odnos glumaca koji su osvojili Oskara po godištu')
    plt.show()

    total_per_quarter = merged_df[
        ['Genre_action', 'Genre_biography', 'Genre_crime', 'Genre_comedy', 'Genre_drama', 'Genre_fantasy',
         'Genre_sci-fi',
         'Genre_mystery', 'Genre_music', 'Genre_romance', 'Genre_history', 'Genre_war', 'Genre_thriller',
         'Genre_adventure',
         'Genre_family']].sum()
    plt.figure(figsize=(16, 12))
    plt.pie(total_per_quarter, labels=total_per_quarter.index, autopct='%1.1f%%', startangle=90)
    plt.title('Broj glumaca koji su nominovani za Oskara po žanrovima')
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=merged_df, x='Oscarstat_previousnominations_acting', hue='Winner')
    plt.title('Distribucija prethodnih nominacija za Oskara u odnosu na osvajanje Oskara')
    plt.xlabel('Broj prethodnih nominacija')
    plt.ylabel('Broj glumaca')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvojen'])

    plt.subplot(2, 2, 2)
    sns.countplot(data=merged_df, x='Oscarstat_previouswins_acting', hue='Winner')
    plt.title('Distribucija prethodnih osvajanja Oskara za glumu')
    plt.xlabel('Broj prethodnih pobeda')
    plt.ylabel('Broj glumaca')
    plt.legend(title='Oskar', labels=['Nije osvojen', 'Osvojen'])
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    sns.boxplot(data=merged_df, x='Winner', y='Birthyear')
    plt.title('Distribucija godina rođenja glumaca u odnosu na osvajanje Oskara')
    plt.xlabel('Oskar')
    plt.ylabel('Godina rođenja')
    plt.xticks([0, 1], ['Nije osvojen', 'Osvojen'])

    plt.subplot(2, 2, 2)
    sns.boxplot(data=merged_df, x='Winner', y='Age')
    plt.title('Godine osvajanja Oskara u odnosu na pobednike')
    plt.xlabel('Osvajanje Oskara (0 = Ne, 1 = Da)')
    plt.ylabel('Godine')
    plt.show()
    plt.show()

