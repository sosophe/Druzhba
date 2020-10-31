from pymystem3 import Mystem
from nltk.corpus import stopwords

russian_sw = set(stopwords.words('russian'))

def tag_mystem(text='Текст нужно передать функции в виде строки!'):  
    m = Mystem()
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
            tagged.append(lemma.lower() + '_' + pos)
        except KeyError:
            continue # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
    return tagged