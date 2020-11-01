import pandas as pd
import numpy as np
import seaborn as sns
from pymystem3 import Mystem
import nltk
import requests
import re

nltk.download("stopwords")
from nltk.corpus import stopwords

russian_stopwords = stopwords.words("russian")


def tag_mystem(text='Текст нужно передать функции в виде строки!', mapping):  
    m = Mystem()
    processed = m.analyze(text)
    tagged = []
    for w in processed:
        try:
            if w["analysis"]:
                lemma = w["analysis"][0]["lex"].lower().strip()
                pos = w["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
    #             print(lemma)
                if lemma not in set(russian_stopwords):
                    if pos in mapping:
                        tagged.append(lemma + '_' + mapping[pos]) # здесь мы конвертируем тэги
                    else:
                        tagged.append(lemma + '_X') # на случай, если попадется тэг, которого нет в маппинге
            else:
                continue
        except KeyError:
            continue # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
    return tagged


def get_words_embed(name, model, mapping):
    res = []
    stems = tag_mystem(text=name)
    for word in stems:
        try:
            res.append(model.get_vector(word))
        except:
            print(word)
            continue
    return res


def match_age_cat(text):
    if text == "0+":
        return [1, 1, 1, 1, 1]
    elif text == "6+":
        return [0, 1, 1, 1, 1]
    
    elif text == "12+":
        return [0, 0, 1, 1, 1]
    
    elif text == "16+":
        return [0, 0, 0, 1, 1]
    else:
        return [0, 0, 0, 0, 1]


def compose_embedd_vector(words, age):
    """
    Example:
    
    > words = np.array([[1, 2, 3], [-1, 0, 13], [0, 2, -3]])
    > array([[ 1,  2,  3],
             [-1,  0, 13],
             [ 0,  2, -3]])
           
    > age = np.array([1, 1, 1, 0, 0])
    > array([1, 1, 1, 0, 0])
    
    > compose_embedd_vector(words, age)
    > array([-1,  0, -3,  1,  2, 13,  1,  1,  1,  0,  0])
    """
    min_vec = words.min(axis=0)
    max_vec = words.max(axis=0)
    return np.concatenate((min_vec[0], max_vec[0], np.array(age)), axis=0)


def get_top_workshops(interest, age_category, df_cats, model, mapping, top=10):
    categories = df_cats.copy()
    embeddings = []
    age_category = np.array(match_age_cat(age_category))
    for word in interest:
        embeddings.append(get_words_embed(word, word_model, mapping))
    average_embedding = compose_embedd_vector(np.array(embeddings), np.array(age_category))
    all_vectors = df_cats.iloc[:, 1:].values
    categories['similarity'] = word_model.cosine_similarities(average_embedding, all_vectors)
    return ((categories.sort_values(by=['similarity'], ascending=False)).name[:10].values.tolist())



def get_club_recommendations(list_of_interests, age, topN=10,
                            word_model_file='word_model.pkl',
                            club_categories_embedding_file='cats_embed.pkl',
                            master_clubs_file='кружки.csv'
                            ):
    word_model = pd.read_pickle(word_model_file)
    url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'
    mapping = {}
    r = requests.get(url, stream=True)
    for pair in r.text.split('\n'):
        pair = re.sub('\s+', ' ', pair, flags=re.U).split(' ')
        if len(pair) > 1:
            mapping[pair[0]] = pair[1]

    df_cats = pd.read_pickle(club_categories_embedding_file).T.reset_index().rename(columns={'index':'name'})

    workshops = get_top_workshops(list_of_interests, "12+", df_cats, word_model, mapping, top=topN)
    df_master = pd.read_csv('кружки.csv')
    df_master['visited'] = 1
    df_ids = df_master[df_master.Наименование.isin(workshops)].id_ученика.unique()
    df_users = df_master[df_master.id_ученика.isin(df_ids)].pivot_table(index = 'id_ученика', columns = 'Наименование', values = 'visited').fillna(0)
    group_corrs = df_users.corr(method = 'pearson', min_periods = 80)
    return group_corrs.sum().sort_values()[-topN:]

