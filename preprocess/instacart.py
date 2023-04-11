import pandas as pd
import random
'''
Reads the raw files, renames columns, last basket as test and the rest as train.
No additional preprocessing steps.
'''
min_basket_per_user = 4
min_item_per_basket = 4
sample_size = 30000

prior_orders_file_path = '/Users/mozhdeh/Data/instacart_2017_05_01/order_products__prior.csv'
train_orders_file_path = '/Users/mozhdeh/Data/instacart_2017_05_01/order_products__train.csv'
orders_file_path = '/Users/mozhdeh/Data/instacart_2017_05_01/orders.csv'
train_baskets_file_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/train_baskets.csv'
test_baskets_file_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/test_baskets.csv'
valid_baskets_file_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/valid_baskets.csv'

prior_orders = pd.read_csv(prior_orders_file_path)
train_orders = pd.read_csv(train_orders_file_path)
all_orders = pd.concat([prior_orders,train_orders])

order_info = pd.read_csv(orders_file_path)

all_orders = pd.merge(order_info,all_orders,how='inner')
print('all_orders')
print(all_orders.shape)
print(all_orders.nunique())


all_orders = all_orders.rename(columns={'order_id':'basket_id', 'product_id':'item_id'})
item_per_basket = all_orders[['item_id','basket_id']].drop_duplicates() \
    .groupby('basket_id').agg({'item_id':'count'}).reset_index()
item_per_basket = item_per_basket[item_per_basket['item_id'] >= min_item_per_basket]
baskets = set(item_per_basket['basket_id'].tolist())
all_orders = all_orders[all_orders['basket_id'].isin(baskets)]
basket_per_user = all_orders[['user_id','basket_id']].drop_duplicates() \
    .groupby('user_id').agg({'basket_id':'count'}).reset_index()
basket_per_user = basket_per_user[basket_per_user['basket_id'] >= min_basket_per_user]
all_users = set(basket_per_user['user_id'].tolist())
####
users = random.sample(all_users,sample_size)
####
all_orders = all_orders[all_orders['user_id'].isin(users)]
print('processed all_orders')
print(all_orders.shape)
print(all_orders.nunique())


last_baskets = all_orders[['user_id','basket_id','order_number']].drop_duplicates() \
    .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'order_number'))
last_baskets.index = last_baskets.index.droplevel()
test_baskets = pd.merge(last_baskets, all_orders, how='left')
train_baskets = pd.concat([all_orders,test_baskets]).drop_duplicates(keep=False)

last_baskets = train_baskets[['user_id','basket_id','order_number']].drop_duplicates() \
    .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'order_number'))
last_baskets.index = last_baskets.index.droplevel()
valid_baskets = pd.merge(last_baskets, all_orders, how='left')
train_baskets = pd.concat([train_baskets,valid_baskets]).drop_duplicates(keep=False)

print('train')
print(train_baskets.shape)
print(train_baskets.nunique())
print('valid')
print(valid_baskets.shape)
print(valid_baskets.nunique())
print('test')
print(test_baskets.shape)
print(test_baskets.nunique())

train_baskets.to_csv(train_baskets_file_path,index=False)
test_baskets.to_csv(test_baskets_file_path,index=False)
valid_baskets.to_csv(valid_baskets_file_path,index=False)