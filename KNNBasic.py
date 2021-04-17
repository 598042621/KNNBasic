import os
import pickle
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic

user_file_path = r'./dataset/dataset1/user.csv'
movie_file_path = r'./dataset/dataset1/movie.csv'
user_item_rating_dump_path = r'./dataset/cache/user_item_rating.pkl'
movie_name_id_dump_path = r'./dataset/cache/movie_name_id.pkl'
rid_to_name_dump_path = r'./dataset/cache/rid_to_name.pkl'
name_to_rid_dump_path = r'./dataset/cache/name_to_rid.pkl'

# 1. 数据预处理，输出user_item_rating
def DataPreprocessing():
    print('Start Data Pre-process...\n')
    if os.path.exists(user_item_rating_dump_path):
        user_item_rating = pickle.load(open(user_item_rating_dump_path, 'rb'))
        if os.path.exists(movie_name_id_dump_path):
            movie_name_id = pickle.load(open(movie_name_id_dump_path, 'rb'))
        else:
            raise Exception('movie_name_id is not exist')
    else:
        user_data = pd.read_csv(user_file_path)
        movie_data = pd.read_csv(movie_file_path)

        movie_list = movie_data['电影名'].unique().tolist()
        movie_data['movie_id'] = movie_data['电影名'].apply(lambda x: movie_list.index(x))
        #movie_name_id = movie_data[['电影名', 'movie_id']].drop_duplicates()
        movie_name_id = movie_data[['电影名', 'movie_id']]

        user_data.drop(columns=['用户名', '评论时间', '类型'], inplace=True)

        user_item_rating = pd.merge(user_data, movie_name_id, on='电影名')
        user_item_rating.rename(columns={'评分': 'rating', '用户ID': 'userID', 'movie_id': 'itemID'}, inplace=True)

        pickle.dump(user_item_rating, open(user_item_rating_dump_path, 'wb'))
        pickle.dump(movie_name_id, open(movie_name_id_dump_path, 'wb'))

    return user_item_rating, movie_name_id

def read_item_names(user_item_rating):
    if os.path.exists(rid_to_name_dump_path):
        rid_to_name = pickle.load(open(rid_to_name_dump_path, 'rb'))
        name_to_rid = pickle.load(open(name_to_rid_dump_path, 'rb'))
    else:
        rid_to_name = {}
        name_to_rid = {}
        for i in range(user_item_rating.shape[0]):
            rid_to_name[user_item_rating.iloc[i]['itemID']] = user_item_rating.iloc[i]['电影名']
            name_to_rid[user_item_rating.iloc[i]['电影名']] = user_item_rating.iloc[i]['itemID']

        pickle.dump(rid_to_name, open(rid_to_name_dump_path, 'wb'))
        pickle.dump(name_to_rid, open(name_to_rid_dump_path, 'wb'))

    return rid_to_name, name_to_rid

def getSimModel(user_item_rating):
    print('Start Creat Model...\n')
    reader = Reader(rating_scale=(2,10))
    data = Dataset.load_from_df(user_item_rating[['userID', 'itemID', 'rating']], reader)

    sim_options = {'name': 'pearson_baseline',
                   'user_based': False  # 计算物品间的相似度, False会使得algo.sim返回对角线矩阵
                   }
    algo = KNNBasic(sim_options = sim_options)

    trainset = data.build_full_trainset()

    algo.fit(trainset)

    return algo

# 基于之前训练的模型 进行相关电影的推荐  步骤：3
def showSimilarMovies(algo, rid_to_name, name_to_rid, movie_name, k):
    # 获得电影Toy Story (1995)的raw_id
    toy_story_raw_id = name_to_rid[movie_name]
    #把电影的raw_id转换为模型的内部id
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
    #通过模型获取推荐电影 这里设置的是10部
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k)
    #模型内部id转换为实际电影id
    neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors]
    #通过电影id列表 或得电影推荐列表
    neighbors_movies = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
    print("The 10 nearest neighbors of "+ movie_name+" are:")
    for movie in neighbors_movies:
        print(movie)


if __name__ == '__main__':
    # 数据预处理，返回user_item_rating用于模型训练, 返回movie_name_id用于ID与名称转换
    user_item_rating, movie_name_id = DataPreprocessing()
    rid_to_name, name_to_rid = read_item_names(user_item_rating)

    # 训练模型
    algo = getSimModel(user_item_rating)

    # 召回与目标电影相似的Top10
    showSimilarMovies(algo, rid_to_name, name_to_rid, '我不是药神', 10)