import pandas as pd
import numpy as np

# Библиотеки

path_to_lib_data = 'Датасеты/'
df_events = pd.read_csv(path_to_lib_data+"Мероприятия - ЭНБ Москвы.csv")
df_catalog = pd.read_csv(path_to_lib_data+"Каталог.csv")
df_readers = pd.read_csv(path_to_foldpath_to_lib_dataers+"Читатели.csv")

ex = pd.ExcelFile('Экземпляры.xlsx')
df_ex = pd.DataFrame()
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet)
    df_ex = df_ex.append(df)
ex = pd.ExcelFile('Экземпляры_2.xlsx')
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet)
    df_ex = df_ex.append(df)
pd.to_csv("Экземпляры.csv")

ex = pd.ExcelFile('Выдача_1.xlsx')
df_give = pd.DataFrame()
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet)
    df_give = df_give.append(df)
ex = pd.ExcelFile('Выдача_2.xlsx')
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet)
    df_give = df_give.append(df)
pd.to_csv("Выдача.csv")


df_ex = df_ex.merge(df_catalog, left_on=['ИД Каталожной записи'], right_on='doc_id')
df_vyd = df_vyd.merge(df_ex, on='Штрих-код')
df_vyd['rating'] = 1
def update_weight(group):
    weights = group['p650a'].value_counts(normalize=True).to_dict()
    group['rating'] = group['p650a'].map(weights) 
    return group
df_vyd = df_vyd.groupby('ИД читателя').progress_apply(update_weight)
df_vyd[['ИД выдачи', 'ИД читателя', 'Инвентарный номер_x',
       'Штрих-код', 'Идентификатор экземпляра', 'ИД Каталожной записи',
       'doc_id', 'p100a', 'p245a', 'p650a', 'p521a', 'rating']].to_csv('vydacha_rated.csv')


# Кружки
path_to_club_data="./КДФ/"

df_class = pd.read_csv(path_to_club_data+"Classificator_hachaton.csv", sep=';')
df_accepted = pd.read_csv(path_to_club_data+"MegaRelation_hackaton.csv", sep=';')
df_member = pd.read_csv(path_to_club_data+"Pupil_hackaton.csv", sep=';')
df_org = pd.read_csv(path_to_club_data+"org_hackaton.csv", sep=';')
df_request = pd.read_csv(path_to_club_data+"request_hackaton.csv", sep=';')
df_services = pd.read_csv(path_to_club_data+"services_hackaton.csv", sep=';')

id_name_dict = dict(zip(df_class.id_классификатора, df_class.Наименование))
parent_dict = dict(zip(df_class.id_классификатора, df_class.id_родительского_классификатора))

def find_parent(x):
    value = parent_dict.get(x, None)
    if value is None:
        return ""
    else:
        # Incase there is a id without name.
        if id_name_dict.get(value, None) is None:
            return "" + find_parent(value)

        return str(id_name_dict.get(value)) +";"+ find_parent(value)
    
def find_parent_category(x):
    res = find_parent(x)
    print(res)
    if 'Хоровое' in res:
        print(res)
    if len(res) == 0:
        return x
    res = res.split(';')
    if len(res) == 1:
        return res[0]
    if res[-1] == '':
        res = res[:-1]
    if len(res) == 1:
        return res[0]
    return res[-2]
    
df_class['Tag'] = df_class.id_классификатора.apply(lambda x: find_parent_category(x)).str.rstrip(';')

df_master = pd.merge(df_member, df_accepted, on='id_ученика', suffixes=('', '_y'))
df_master.drop(df_master.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
df_master = pd.merge(df_master, df_services, on='id_услуги', suffixes=('', '_y'))
df_master.drop(df_master.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
df_master = pd.merge(df_master, df_org, on='id_организации', suffixes=('', '_y'))
df_master.drop(df_master.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
df_master = pd.merge(df_master, df_org, on='id_организации', suffixes=('', '_y'))
df_master.drop(df_master.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
# df_master = pd.merge(df_master, df_request, on='id_услуги', suffixes=('', '_y'))
# df_master.drop(df_master.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
df_master = pd.merge(df_master, df_class, left_on='Классификатор_услуги', right_on='id_классификатора', suffixes=('', '_y'))
df_master.drop(df_master.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
df_master.rename(columns={'Tag':"Наименование_родителя"}, inplace=True)
df_master['Наименование_родителя'] = df_master['Наименование_родителя'].fillna(df_master['Наименование'])

df_master.to_csv("кружки.csv")
