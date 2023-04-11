import numpy as np
import pandas as pd
from scipy import sparse
import similaripy as sim
import pickle

train_basket_path = 'data/instacart_30k/train_baskets.csv'
train_baskets = pd.read_csv(train_basket_path)

baskets_df = train_baskets[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
    .apply(list).reset_index(name='items')
basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['items']))

user_baskets_df = train_baskets[['basket_id','user_id']].drop_duplicates()
user_baskets = user_baskets_df.groupby(['user_id'])['basket_id'].apply(list) \
    .reset_index(name='baskets')
user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['baskets']))

item_base_scores = {}
for user in user_baskets_dict:
    baskets = user_baskets_dict[user]
    basket_len = len(baskets)
    if user not in item_base_scores:
        item_base_scores[user] = {}
        for basket_index,basket in enumerate(baskets):
            w1_b = 1./float(basket_len - basket_index)
            for item in basket_items_dict[basket]:
                if item not in item_base_scores[user]:
                    item_base_scores[user][item] = 0
                item_base_scores[user][item] += w1_b
data_list = []
for user in item_base_scores:
    baskets = user_baskets_dict[user]
    basket_len = len(baskets)
    for item in item_base_scores[user]:
        score = float(item_base_scores[user][item]) / float(basket_len)
        data_list.append([user, item, score])

df = pd.DataFrame(data_list, columns = ['user', 'item','score'])
df.to_csv('data/instacart_30k/user_item_scores.csv',index=False)

df = pd.read_csv('data/instacart_30k/user_item_scores.csv')


df_users = set(df['user'].tolist())
df_items = set(df['item'].tolist())
item_dic = {}
rev_item_dic = {}
for i,item in enumerate(df_items):
    item_dic[item] = i
    rev_item_dic[i] = item
user_dic = {}
rev_user_dic = {}
for i,user in enumerate(df_users):
    user_dic[user] = i
    rev_user_dic[i] = user

df['uid'] = df['user'].apply(lambda x: user_dic[x])
df['pid'] = df['item'].apply(lambda x: item_dic[x])

n_users = len(set(df['user'].tolist()))
n_items = len(set(df['item'].tolist()))
userItem_mat = sparse.coo_matrix(( df.score.values  , (df.uid.values, df.pid.values)), shape=(n_users,n_items))
userSim = sim.asymmetric_cosine(sparse.csr_matrix(userItem_mat), alpha=0.5, k=50)
user_sim_dict = dict(userSim.todok().items()) # convert to dictionary of keys format
final_user_sim_dict = {}
for key in user_sim_dict:
    final_user_sim_dict[(rev_user_dic[key[0]],rev_user_dic[key[1]])] = user_sim_dict[key]
with open('data/instacart_30k/user_sim.pickle', 'wb') as handle:
    pickle.dump(final_user_sim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)