from random import sample
from math import sqrt
import pickle
import numpy as np


class Collab:
    def __init__(self,train_baskets,test_samples):
        self.train_baskets = train_baskets
        self.test_samples = test_samples
        self.basket_items_dict = {}
        self.user_baskets_dict = {}
        self.user_sim_dict = {}
        self.user_neighbors = {}

    def train(self):
        baskets_df = self.train_baskets[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
        basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
            .apply(list).reset_index(name='items')
        self.basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['items']))

        user_baskets_df = self.train_baskets[['basket_id','user_id']].drop_duplicates()
        user_baskets = user_baskets_df.groupby(['user_id'])['basket_id'].apply(list) \
            .reset_index(name='baskets')
        self.user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['baskets']))

        with open('data/instacart/user_sim.pickle', 'rb') as handle:
            self.user_sim_dict = pickle.load(handle)

        for key in self.user_sim_dict:
            if key[0] not in self.user_neighbors:
                self.user_neighbors[key[0]] = []
            self.user_neighbors[key[0]].append(key[1])

    def user_predictions(self,user, input_items):
        baskets = self.user_baskets_dict[user]
        basket_len = len(baskets)

        item_base_scores = {}
        for basket_index,basket in enumerate(baskets):
            w1_b = 1./float(basket_len - basket_index)
            for item in self.basket_items_dict[basket]:
                if item not in item_base_scores:
                    item_base_scores[item] = 0
                item_base_scores[item] += w1_b

        current_scores = {}
        current_items_len = len(input_items)
        for current_item_index, current_item in enumerate(input_items):
            w2_j = 1./float(current_items_len - current_item_index)
            for basket_index,basket in enumerate(baskets):
                if current_item in self.basket_items_dict[basket]:
                    w1_b = 1./float(basket_len - basket_index)
                    i_index = self.basket_items_dict[basket].index(current_item)
                    for j_index,item in enumerate(self.basket_items_dict[basket]):
                        if i_index == j_index:
                            continue
                        w3_ij = 1./float(abs(i_index - j_index))
                        if item not in current_scores:
                            current_scores[item] = 0
                        current_scores[item] += w3_ij * w1_b * w2_j

        alpha1 = 1
        alpha2 = np.log(current_items_len)
        final_item_scores = {}
        for item in item_base_scores:
            final_item_scores[item] = alpha1 * item_base_scores[item]
            if item in current_scores:
                final_item_scores[item] += alpha2 * current_scores[item]

        return final_item_scores

    def predict(self):
        test_inputs = self.test_samples['input_items'].apply(eval).tolist()
        test_users = self.test_samples['user_id'].tolist()

        predictions = []

        for i, input_items in enumerate(test_inputs):
            if i% 100000 == 0:
                print(i)
            user = test_users[i]

            personal_scores = self.user_predictions(user,input_items)
            neighbor_scores = {}
            for neighbor in self.user_neighbors[user]:
                scores = self.user_predictions(neighbor,input_items)
                neighbor_scores[neighbor] = scores

            agg_neighbor_scores = {}
            norm_term = {}
            for neighbor in neighbor_scores:
                sim = self.user_sim_dict[(user,neighbor)]
                item_scores = neighbor_scores[neighbor]
                for item in item_scores:
                    if item not in agg_neighbor_scores:
                        agg_neighbor_scores[item] = 0
                        norm_term[item] = 0
                    agg_neighbor_scores[item] += sim * item_scores[item]
                    norm_term[item] += 1

            beta1 = 1
            beta2 = 1
            final_item_scores = {}
            for item in personal_scores:
                final_item_scores[item] = beta1 * personal_scores[item]
            for item in agg_neighbor_scores:
                if item not in final_item_scores:
                    final_item_scores[item] = 0
                final_item_scores[item] += beta2 * (agg_neighbor_scores[item]/norm_term[item])

            sorted_item_scores = sorted(final_item_scores.items(),key= lambda x:x[1], reverse=True)
            predicted_items = [x[0] for x in sorted_item_scores]
            predictions.append(predicted_items)
        return predictions,predictions



