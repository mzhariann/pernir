import numpy as np
from models.tifuknn import TIFUKNN
from models.global_oracle import GlobalOracle
from models.local_oracle import LocalOracle
from models.sknn import SKNN
from models.vsknn import VSKNN
from models.stan import STAN
from models.psknn import PSKNN
from models.next_item import NextItem
from models.collab import Collab
from models.ppop import PPop
import pandas as pd
from metrics import *
import pickle

train_basket_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/train_baskets.csv'
test_sample_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/test_samples.csv'

train_baskets = pd.read_csv(train_basket_path)
test_samples = pd.read_csv(test_sample_path)
'''
item_users_df = train_baskets[['user_id','item_id']].drop_duplicates()
item_users = item_users_df.groupby(['item_id'])['user_id'].apply(list) \
    .reset_index(name='users')
item_users_dict = dict(zip(item_users['item_id'],item_users['users']))

lens = []
user_neighbors = {}
print(len(item_users_dict))
for i,item in enumerate(item_users_dict):
    if i % 100 == 0:
        print(i)
    for user in item_users_dict[item]:
        if user not in user_neighbors:
            user_neighbors[user] = {}
        for user2 in item_users_dict[item]:
            key = str(user)+'_'+str(user2)
            if key not in user_neighbors:
                user_neighbors[key] = []
            user_neighbors[key].append(item)

with open('/Users/mozhdeh/PycharmProjects/psbr/data/instacart/user_neighbors.pickle', 'wb') as handle:
    pickle.dump(user_neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''


model = Collab(train_baskets,test_samples)
print('train...')
model.train()
print('predict...')
predicted_single_labels = model.predict()
result_df = pd.DataFrame(columns=['predicted_single_labels'])
result_df['predicted_single_labels'] = predicted_single_labels
result_df.to_csv('/Users/mozhdeh/PycharmProjects/psbr/results/instacart_pstan.csv',index=False)

result_df = pd.read_csv('/Users/mozhdeh/PycharmProjects/psbr/results/instacart_pstan.csv')
predicted_single_labels = result_df['predicted_single_labels']
test_label = test_samples['label_item'].tolist()
test_labels = test_samples['label_items']#.apply(eval).tolist()
test_inputs = test_samples['input_items'].tolist()
k_set = [10,20]
hr = []
p = []
r = []
mrr = []
ndcg = []
for k in k_set:
    print('k:',k)
    hr = []
    mrr = []
    for i in range(len(test_label)):
        if i%100000 == 0:
            print(i)
        label = test_label[i]
        test_input = eval(test_inputs[i])
        labels = eval(test_labels[i])
        _pred = eval(predicted_single_labels[i]) #eval(predicted_single_labels[i])
        pred = []
        #preds = []
        for item in _pred:
            if item not in test_input:
                pred.append(item)
        preds = pred
        if k == 'B':
            k = len(labels)
        hr.append(hr_k(label,pred,k))
        mrr.append(mrr_k(label,pred,k))
        #p.append(precision_k(labels,preds,k))
        #r.append(recall_k(labels,preds,k))
        #ndcg.append(ndcg_k(labels,preds,k))
    print('hr:',np.mean(hr))
    print('mrr:',np.mean(mrr))
    #print('recall:',np.mean(r))
    #print('precision:',np.mean(p))
    #print('ndcg:',np.mean(ndcg))


