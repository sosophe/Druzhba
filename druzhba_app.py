#!/usr/bin/env python
# coding: utf-8

# In[8]:


import collections
import functools
import pickle
import string
from multiprocessing import Pool

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# import pickle
import re
import sys
import zipfile

import gensim
import nltk
import requests
import seaborn as sns
from nltk.corpus import stopwords
from pymystem3 import Mystem





# In[87]:
@st.cache(show_spinner=False)
def import_database():
    database = pd.read_csv("database.csv")
    books = database.itemid.values.copy()
    return database, books


@st.cache(show_spinner=False)
def import_events():
    path = "../final_data/Мероприятия.csv"
    events = pd.read_csv(
        path,
        usecols=[
            "Название мероприятия",
            "Тип мероприятия",
            "Направленность мероприятия",
            "Краткое описание",
            "Округ",
            "Район",
            "Возрастной ценз участников мероприятия",
            "Возрастная категория",
        ],
    )
    return events

@st.cache(show_spinner=False)
def import_event_type_embedding():
    event_type_embedding = pickle.load(open("event_type_embedding.pickle", "rb"))
    return event_type_embedding

# @st.cache(show_spinner=False)
# def import_merged():
#     merged = pd.merge(
#         events,
#         event_type_embedding,
#         on="Направленность мероприятия",
#         suffixes=("", "_y"),
#     )
#     return merged

@st.cache(show_spinner=False)
def import_word_model():
    model_file = "180.zip"  # model_url.split('/')[-1]
    with zipfile.ZipFile(model_file, "r") as archive:
        stream = archive.open("model.bin")
        word_model = gensim.models.KeyedVectors.load_word2vec_format(
            stream, binary=True
        )
    return word_model

# @st.cache(show_spinner=False)
def import_rec_model():
    algo = pickle.load(open("recom_model.pickle", "rb"))
    return algo

@st.cache(show_spinner=False)
def import_raw_dict():
    with open("raw_dict.pickle", "rb") as pickle_in:
        cat_to_vec = pickle.load(pickle_in)
        return cat_to_vec

@st.cache(show_spinner=False)
def import_readers_birthday():
    readers_birthday = pd.read_csv("readers_birthday.csv")
    return readers_birthday

@st.cache(show_spinner=False)
def import_cluster_embedding():
    cluster_embedding = pickle.load(open("cluster_embedding.pickle", "rb"))
    return cluster_embedding


@st.cache(show_spinner=False)
def import_cluster_embedd_data():
    cluster_embedd_data = pickle.load(open("cluster_embedd_data.pickle", "rb"))
    return cluster_embedd_data

@st.cache(show_spinner=False)
def import_cluster_embedd_data():
    clean_description_data = pickle.load(open("clean_description.pickle", "rb"))
    return clean_description_data


@st.cache(show_spinner=False)
def import_data():
    table = str.maketrans("", "", string.punctuation)
    return table


@st.cache(show_spinner=False)
def import_stemmer():
    stemmer = SnowballStemmer("russian")
    return stemmer

# @st.cache(show_spinner=False)
# def import_russian_stopwords():
#     russian_stopwords = stopwords.words("russian")
#     return russian_stopwords


# @st.cache(show_spinner=False)
def get_predictoins(input_iid, input_uid):
    return algo.predict(uid=input_uid, iid=input_iid, verbose=False).est

# @st.cache(show_spinner=False)
def get_books(input_uid):
    with Pool(15) as p:
        pred_rating = list(
            p.map(functools.partial(get_predictoins, input_uid=input_uid), books)
        )

    pred_book = [(rating, book) for rating, book in zip(pred_rating, books)]

    final_pred_books = [
        x[1] for x in sorted(pred_book, key=lambda x: x[0], reverse=True)[:30]
    ]
    try:
        user_age = (
            readers_birthday[readers_birthday.userid == input_uid]["age"]
        ).values[0]
    except Exception as e:
        user_age = 18

    df = database[database.itemid.isin(final_pred_books)]

    return df[(df.age_cat <= user_age)], user_age


russian_stop_words = {
    "words": [
        "а",
        "е",
        "и",
        "ж",
        "м",
        "о",
        "на",
        "не",
        "ни",
        "об",
        "но",
        "он",
        "мне",
        "мои",
        "мож",
        "она",
        "они",
        "оно",
        "мной",
        "много",
        "многочисленное",
        "многочисленная",
        "многочисленные",
        "многочисленный",
        "мною",
        "мой",
        "мог",
        "могут",
        "можно",
        "может",
        "можхо",
        "мор",
        "моя",
        "моё",
        "мочь",
        "над",
        "нее",
        "оба",
        "нам",
        "нем",
        "нами",
        "ними",
        "мимо",
        "немного",
        "одной",
        "одного",
        "менее",
        "однажды",
        "однако",
        "меня",
        "нему",
        "меньше",
        "ней",
        "наверху",
        "него",
        "ниже",
        "мало",
        "надо",
        "один",
        "одиннадцать",
        "одиннадцатый",
        "назад",
        "наиболее",
        "недавно",
        "миллионов",
        "недалеко",
        "между",
        "низко",
        "меля",
        "нельзя",
        "нибудь",
        "непрерывно",
        "наконец",
        "никогда",
        "никуда",
        "нас",
        "наш",
        "нет",
        "нею",
        "неё",
        "них",
        "мира",
        "наша",
        "наше",
        "наши",
        "ничего",
        "начала",
        "нередко",
        "несколько",
        "обычно",
        "опять",
        "около",
        "мы",
        "ну",
        "нх",
        "от",
        "отовсюду",
        "особенно",
        "нужно",
        "очень",
        "отсюда",
        "в",
        "во",
        "вон",
        "вниз",
        "внизу",
        "вокруг",
        "вот",
        "восемнадцать",
        "восемнадцатый",
        "восемь",
        "восьмой",
        "вверх",
        "вам",
        "вами",
        "важное",
        "важная",
        "важные",
        "важный",
        "вдали",
        "везде",
        "ведь",
        "вас",
        "ваш",
        "ваша",
        "ваше",
        "ваши",
        "впрочем",
        "весь",
        "вдруг",
        "вы",
        "все",
        "второй",
        "всем",
        "всеми",
        "времени",
        "время",
        "всему",
        "всего",
        "всегда",
        "всех",
        "всею",
        "всю",
        "вся",
        "всё",
        "всюду",
        "г",
        "год",
        "говорил",
        "говорит",
        "года",
        "году",
        "где",
        "да",
        "ее",
        "за",
        "из",
        "ли",
        "же",
        "им",
        "до",
        "по",
        "ими",
        "под",
        "иногда",
        "довольно",
        "именно",
        "долго",
        "позже",
        "более",
        "должно",
        "пожалуйста",
        "значит",
        "иметь",
        "больше",
        "пока",
        "ему",
        "имя",
        "пор",
        "пора",
        "потом",
        "потому",
        "после",
        "почему",
        "почти",
        "посреди",
        "ей",
        "два",
        "две",
        "двенадцать",
        "двенадцатый",
        "двадцать",
        "двадцатый",
        "двух",
        "его",
        "дел",
        "или",
        "без",
        "день",
        "занят",
        "занята",
        "занято",
        "заняты",
        "действительно",
        "давно",
        "девятнадцать",
        "девятнадцатый",
        "девять",
        "девятый",
        "даже",
        "алло",
        "жизнь",
        "далеко",
        "близко",
        "здесь",
        "дальше",
        "для",
        "лет",
        "зато",
        "даром",
        "первый",
        "перед",
        "затем",
        "зачем",
        "лишь",
        "десять",
        "десятый",
        "ею",
        "её",
        "их",
        "бы",
        "еще",
        "при",
        "был",
        "про",
        "процентов",
        "против",
        "просто",
        "бывает",
        "бывь",
        "если",
        "люди",
        "была",
        "были",
        "было",
        "будем",
        "будет",
        "будете",
        "будешь",
        "прекрасно",
        "буду",
        "будь",
        "будто",
        "будут",
        "ещё",
        "пятнадцать",
        "пятнадцатый",
        "друго",
        "другое",
        "другой",
        "другие",
        "другая",
        "других",
        "есть",
        "пять",
        "быть",
        "лучше",
        "пятый",
        "к",
        "ком",
        "конечно",
        "кому",
        "кого",
        "когда",
        "которой",
        "которого",
        "которая",
        "которые",
        "который",
        "которых",
        "кем",
        "каждое",
        "каждая",
        "каждые",
        "каждый",
        "кажется",
        "как",
        "какой",
        "какая",
        "кто",
        "кроме",
        "куда",
        "кругом",
        "с",
        "т",
        "у",
        "я",
        "та",
        "те",
        "уж",
        "со",
        "то",
        "том",
        "снова",
        "тому",
        "совсем",
        "того",
        "тогда",
        "тоже",
        "собой",
        "тобой",
        "собою",
        "тобою",
        "сначала",
        "только",
        "уметь",
        "тот",
        "тою",
        "хорошо",
        "хотеть",
        "хочешь",
        "хоть",
        "хотя",
        "свое",
        "свои",
        "твой",
        "своей",
        "своего",
        "своих",
        "свою",
        "твоя",
        "твоё",
        "раз",
        "уже",
        "сам",
        "там",
        "тем",
        "чем",
        "сама",
        "сами",
        "теми",
        "само",
        "рано",
        "самом",
        "самому",
        "самой",
        "самого",
        "семнадцать",
        "семнадцатый",
        "самим",
        "самими",
        "самих",
        "саму",
        "семь",
        "чему",
        "раньше",
        "сейчас",
        "чего",
        "сегодня",
        "себе",
        "тебе",
        "сеаой",
        "человек",
        "разве",
        "теперь",
        "себя",
        "тебя",
        "седьмой",
        "спасибо",
        "слишком",
        "так",
        "такое",
        "такой",
        "такие",
        "также",
        "такая",
        "сих",
        "тех",
        "чаще",
        "четвертый",
        "через",
        "часто",
        "шестой",
        "шестнадцать",
        "шестнадцатый",
        "шесть",
        "четыре",
        "четырнадцать",
        "четырнадцатый",
        "сколько",
        "сказал",
        "сказала",
        "сказать",
        "ту",
        "ты",
        "три",
        "эта",
        "эти",
        "что",
        "это",
        "чтоб",
        "этом",
        "этому",
        "этой",
        "этого",
        "чтобы",
        "этот",
        "стал",
        "туда",
        "этим",
        "этими",
        "рядом",
        "тринадцать",
        "тринадцатый",
        "этих",
        "третий",
        "тут",
        "эту",
        "суть",
        "чуть",
        "тысяч",
    ]
}


# In[4]:

# @st.cache(show_spinner=False)
def tokenize_text(text):
    words = text.split()
    # remove punctuation from each word
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table).lower() for w in words]
    removed = [
        word for word in stripped if word not in set(russian_stop_words["words"])
    ]
    stemmed = [stemmer.stem(word) for word in removed]
    return " ".join(stemmed)


# In[5]:

# @st.cache(show_spinner=False)
def get_direction_set(request):
    cl_embd_arr = np.array(
        [element for element in event_type_embedding["embedd_vector"].values]
    )
    X = np.concatenate((cl_embd_arr, request.reshape(-1, 605)))
    #     print(X.shape)
    answer = [(idx, val) for idx, val in enumerate(cosine_similarity(X)[-1])]
    answer = sorted(answer, key=lambda x: x[1], reverse=True)
    answer_cluster = [i[0] for i in answer[1:3]]
    #     print("answer_cluster ", answer_cluster)
    #     print("event_type_embedding.shape = ", event_type_embedding.shape)
    #     print("event_type_embedding.reset_index() ", event_type_embedding.reset_index())
    # idx_in_merged = (
    #     event_type_embedding.reset_index().loc[answer_cluster, :]["index"].values
    # )
    # print("idx_in_merged ", idx_in_merged)
    # directions = np.unique(merged.loc[idx_in_merged, :]["Направленность мероприятия"])
    temp = event_type_embedding.reset_index()
    idx_in_merged = temp[temp.index.isin(answer_cluster)]["index"].values
    #     print("idx_in_merged ", idx_in_merged)
    directions = np.unique(
        merged[merged.index.isin(idx_in_merged)]["Направленность мероприятия"]
    )
    return merged[merged["Направленность мероприятия"].isin(directions)]

# @st.cache(show_spinner=False)
def get_event(embed_vector, request_words, request_age):
    hint_dir = get_direction_set(embed_vector)
    direction = hint_dir.copy()
    direction["low"] = direction["Возрастная категория"].apply(
        lambda x: int(x.split()[1])
    )
    direction["high"] = direction["Возрастная категория"].apply(
        lambda x: int(x.split()[3]) if len(x.split()) == 4 else 999
    )
    l = clean_description_data.iloc[
        direction[
            (direction.high >= request_age) & (direction.low <= request_age)
        ].index
    ].values.tolist()
    tokenized_request = tokenize_text(" ".join(request_words))
    l.append(tokenized_request)
    tfidf = TfidfVectorizer().fit_transform(l)
    cosine_similarities = linear_kernel(tfidf[-1], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-100:-1]
    final_recommendation = list(related_docs_indices)
    final_recommendation.remove(len(l) - 1)
    return events.iloc[final_recommendation]


# ## Рекомендация мероприятия

# In[9]:

# @st.cache(show_spinner=False)
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
    return np.concatenate((min_vec, max_vec, age))





# @st.cache(show_spinner=False)
def tag_mystem(mapping, text="Текст нужно передать функции в виде строки!"):
    m = Mystem()
    processed = m.analyze(text)
    tagged = []
    for w in processed:
        try:
            if w["analysis"]:
                lemma = w["analysis"][0]["lex"].lower().strip()
                pos = w["analysis"][0]["gr"].split(",")[0]
                pos = pos.split("=")[0].strip()
                #             print(lemma)
                if lemma not in set(russian_stopwords):
                    if pos in mapping:
                        tagged.append(
                            lemma + "_" + mapping[pos]
                        )  # здесь мы конвертируем тэги
                    else:
                        tagged.append(
                            lemma + "_X"
                        )  # на случай, если попадется тэг, которого нет в маппинге
            else:
                continue
        except KeyError:
            continue  # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
    return tagged

# @st.cache(show_spinner=False)
def get_words_embed(name, model, mapping):
    res = []
    stems = tag_mystem(text=name, mapping=mapping)
    for word in stems:
        try:
            res.append(model.get_vector(word))
        except:
            print(word)
            continue
    return res

# @st.cache(show_spinner=False)
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

# @st.cache(show_spinner=False)
def get_top_workshops(interest, age_category, df_cats, word_model, mapping, top=10):
    categories = df_cats.copy()
    embeddings = []
    age_category = np.array(match_age_cat(age_category))
    for word in interest:
        embeddings.append(get_words_embed(word, word_model, mapping))
    # average_embedding = compose_embedd_vector_2(
    #     np.array(embeddings), np.array(age_category)
    # )
    average_embedding = compose_embedd_vector_2(embeddings, age_category)
    all_vectors = df_cats.iloc[:, 1:].values
    categories["similarity"] = word_model.cosine_similarities(
        average_embedding, all_vectors
    )
    return (
        (categories.sort_values(by=["similarity"], ascending=False))
        .name[:10]
        .values.tolist()
    )

# @st.cache(show_spinner=False)
def get_club_recommendations(
    list_of_interests,
    age,
    topN=10,
    # word_model_file='word_model.pkl',
    club_categories_embedding_file="cats_embed.csv",
    master_clubs_file="кружки.csv",
):

    url = "https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map"
    mapping = {}
    r = requests.get(url, stream=True)
    for pair in r.text.split("\n"):
        pair = re.sub("\s+", " ", pair, flags=re.U).split(" ")
        if len(pair) > 1:
            mapping[pair[0]] = pair[1]

    # with open(club_categories_embedding_file,"rb") as pickle_in:
    #     v2 = pickle.load(pickle_in)

    df_cats = (
        pd.read_csv(club_categories_embedding_file)
        .T.reset_index()
        .rename(columns={"index": "name"})
    )

    # df_cats = pd.read_pickle(v2).T.reset_index().rename(columns={'index':'name'})

    workshops = get_top_workshops(  # returns list
        list_of_interests, age, df_cats, word_model, mapping, top=topN
    )
    df_master = pd.read_csv(master_clubs_file)
    df_master["visited"] = 1
    df_ids = df_master[df_master.Наименование.isin(workshops)].id_ученика.unique()
    df_users = (
        df_master[df_master.id_ученика.isin(df_ids)]
        .pivot_table(index="id_ученика", columns="Наименование", values="visited")
        .fillna(0)
    )
    group_corrs = df_users.corr(method="pearson", min_periods=80)
    return group_corrs.sum().sort_values().reset_index()[-topN:]

# @st.cache(show_spinner=False)
def compose_embedd_vector_2(words, age):
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
    words_unnest = [l[0] for l in list(words) if len(l) != 0]
    min_vec = np.min(words_unnest, axis=0)
    max_vec = np.max(words_unnest, axis=0)
    return np.concatenate((min_vec, max_vec, np.array(age)), axis=0)

# @st.cache(show_spinner=False)
def get_club_recommendations(
    list_of_interests,
    age,
    topN=10,
    # word_model_file='word_model.pkl',
    club_categories_embedding_file="cats_embed.csv",
    master_clubs_file="кружки.csv",
):

    url = "https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map"
    mapping = {}
    r = requests.get(url, stream=True)
    for pair in r.text.split("\n"):
        pair = re.sub("\s+", " ", pair, flags=re.U).split(" ")
        if len(pair) > 1:
            mapping[pair[0]] = pair[1]

    # with open(club_categories_embedding_file,"rb") as pickle_in:
    #     v2 = pickle.load(pickle_in)

    df_cats = (
        pd.read_csv(club_categories_embedding_file)
        .T.reset_index()
        .rename(columns={"index": "name"})
    )

    # df_cats = pd.read_pickle(v2).T.reset_index().rename(columns={'index':'name'})

    workshops = get_top_workshops(
        list_of_interests, age, df_cats, word_model, mapping, top=topN
    )
    df_master = pd.read_csv(master_clubs_file)
    df_master["visited"] = 1
    df_ids = df_master[df_master.Наименование.isin(workshops)].id_ученика.unique()
    df_users = (
        df_master[df_master.id_ученика.isin(df_ids)]
        .pivot_table(index="id_ученика", columns="Наименование", values="visited")
        .fillna(0)
    )
    group_corrs = df_users.corr(method="pearson", min_periods=80)
    return group_corrs.sum().sort_values().reset_index()[-topN:]


# ## Книги

# In[88]:
# @st.cache(show_spinner=False)
def get_book_recommendation(user_input):
    result_books, user_age = get_books(int(user_input))
    st.subheader(
        "Основываясь на ваших предпочтениях мы рекомендуем вам следующие книги:"
    )
    i = 1
    for _, row in result_books.iterrows():
        author = str(row["author"])
        title = str(row["title"])
        if author == "nan":
            author = ""
        if title == "nan":
            title = ""
        output = f"{i}) " + author + ' "' + title + '"'
        st.write(output)
        i += 1
        if i == 11:
            break
    key_words = result_books["category"].unique()
    request_words = [
        w.translate(table).lower() for w in " ".join(key_words).lower().split()
    ]
    request_input = np.array(
        [cat_to_vec.get(key) for key in key_words if cat_to_vec.get(key) is not None]
    )
    if len(request_input) == 0:
        embed_vector = np.array([0.5 for _ in range(605)])
    else:
        embed_vector = np.mean(request_input, axis=0)

    return (
        result_books,
        user_age,
        key_words,
        request_words,
        request_input,
        embed_vector,
    )


# ## Мероприятия

# In[89]:

# @st.cache(show_spinner=False)
def get_event_recomendations(result_events):
    st.subheader("Возможно вам также будет интересно посетить данные мероприятия:\n")
    i = 1
    for _, row in result_events.iterrows():
        descr = str(row["Краткое описание"])
        title = str(row["Название мероприятия"])
        discrict = str(row["Район"])
        area = str(row["Округ"])
        if descr == "nan":
            descr = ""
        if discrict == "nan":
            discrict = ""
        if area == "nan":
            area = ""
        if title == "nan":
            title = ""
        place = discrict + ", " + area
        age = str(row["Возрастной ценз участников мероприятия"])
        if age == "nan":
            age = ""
        output = f"{i}) " + '"' + title + '"' + ", " + place + " (" + age + ")"
        output2 = f"Краткое описание: \t {descr}\n"
        i += 1
        st.write(output)
        st.write(output2)
        if i == 6:
            break


# ## Кружки

# In[90]:

# @st.cache(show_spinner=False)
def match_cat_age(age):
    if age >= 18:
        return "18+"
    elif age >= 16:
        return "16+"
    elif age >= 12:
        return "12+"
    elif age >= 6:
        return "6+"
    else:
        return "0+"


# In[91]:

# @st.cache(show_spinner=False)
def get_circles_recomendations(request_words, age_request_cat):
    result_circles = get_club_recommendations(
        list_of_interests=np.array(request_words), age=np.array(age_request_cat)
    )
    circle_df = pd.DataFrame(result_circles)
    i = 0
    st.subheader("Рекоммендованные кружки:\n")
    for _, row in circle_df.iterrows():
        i += 1
        title = str(row["Наименование"])
        if title == "nan":
            title = ""
        output = f"{i}) " + title
        st.write(output)


# ## Main call
st.title("Рекомендательная система Дружба")

user_input = st.text_input("Введите пожалуйста ваш id", "")
if user_input != "":
    with st.spinner('Анализируем вашу библиографию...'):
        database, books = import_database()
        events = import_events()
        # merged, event_type_embedding = import_merged()
        word_model = import_word_model()
        algo = import_rec_model()
        cat_to_vec = import_raw_dict()
        readers_birthday = import_readers_birthday()
        cluster_embedding = import_cluster_embedding()
        cluster_embedd_data = import_cluster_embedd_data()
        clean_description_data = import_cluster_embedd_data()
        clean_description_data = import_cluster_embedd_data()
        # russian_stopwords = import_russian_stopwords()
        table = import_data()
        stemmer = import_stemmer()
        event_type_embedding = import_event_type_embedding()
        russian_stopwords = stopwords.words("russian")
        # merged = import_merged()
        merged = pd.merge(
            events,
            event_type_embedding,
            on="Направленность мероприятия",
            suffixes=("", "_y"),
        )



        (
            result_books,
            user_age,
            key_words,
            request_words,
            request_input,
            embed_vector,
        ) = get_book_recommendation(user_input)
    age_request_cat = match_cat_age(user_age)
    # print()
    with st.spinner('Подбираем интересные мероприятия...'):
        result_events = get_event(embed_vector, request_words, user_age)
        get_event_recomendations(result_events)
    # print()
    with st.spinner('Ищем вам новое хобби...'):
        get_circles_recomendations(request_words, age_request_cat)

# In[92]:




# In[ ]:
