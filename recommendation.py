import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_location = 'dataset/Startups.csv'

def get_data():
    startup_data=pd.read_csv(data_location, delimiter=',')
    startup_data['Company'] = startup_data['Company'].str.lower()
    return startup_data

def combine_data(data):
    data_recommend = data.drop(columns=['Satus', 'Year Founded','Mapping Location', 'Founders', 'Y Combinator Year', 'Y Combinator Session', 'Investors', 'Amounts raised in different funding rounds', 'Office Address','Headquarters (City)','Headquarters (US State)', 'Headquarters (Country)', 'Logo', 'Seed-DB / Mattermark Profile','Crunchbase / Angel List Profile','Website'])
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:3]].apply(
                                                                         lambda x: ','.join(x.dropna().astype(str)),axis=1)
    
    data_recommend = data_recommend.drop(columns=['Company', 'Description', 'Categories'])
    return data_recommend


def transform_data(data_combine, data_plot):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['combine'])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_plot['Description'].values.astype('str'))

    combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
    
    cosine_sim = cosine_similarity(combine_sparse, combine_sparse)
    
    return cosine_sim



def recommend_startups(name, data, combine, transform):

    indices = pd.Series(data.index, index = data['Company'])
    index = indices[name]

    sim_scores = list(enumerate(transform[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:5]
    
    startup_indices = [i[0] for i in sim_scores]

    startup_name = data['Company'].iloc[startup_indices]
    startup_categories = data['Categories'].iloc[startup_indices]

    recommendation_data = pd.DataFrame(columns=['Company','Categories'])

    recommendation_data['Company'] = startup_name
    recommendation_data['Categories'] = startup_categories

    return recommendation_data


def results(startup_name):
    startup_name = startup_name.lower()
    
    find_startup = get_data()
    combine_result = combine_data(find_startup)
    transform_result = transform_data(combine_result,find_startup)
    
    if startup_name not in find_startup['Company'].unique():
        return 'Startup not in Database'
    
    else:
        recommendations = recommend_startups(startup_name, find_startup, combine_result, transform_result)
        return recommendations.to_dict('records')