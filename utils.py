from pymystem3 import Mystem
from nltk.corpus import stopwords
import requests
import re

russian_sw = set(stopwords.words('russian'))

url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'

mapping = {}
r = requests.get(url, stream=True)
for pair in r.text.split('\n'):
    pair = re.sub('\s+', ' ', pair, flags=re.U).split(' ')
    if len(pair) > 1:
        mapping[pair[0]] = pair[1]


def tag_mystem(text='Текст нужно передать функции в виде строки!'):  
    m = Mystem()
    #print(text)
    text = ''.join([x for x in text.split(';') if ('итература' not in x )])
    #print(text)
    processed = m.analyze(text)
    tagged = []
    
    for w in processed:
        try:
            if not w["analysis"]: continue
            lemma = w["analysis"][0]["lex"].lower().strip()
            if lemma in russian_sw:
                continue
            pos = w["analysis"][0]["gr"].split(',')[0]
            
            pos = pos.split('=')[0].strip()
            
            if pos in mapping:
                tagged.append(lemma + '_' + mapping[pos]) # здесь мы конвертируем тэги
            else:
                tagged.append(lemma + '_X') # на случай, если попадется тэг, которого нет в маппинге
                
        except KeyError:
            continue
    return tagged, text