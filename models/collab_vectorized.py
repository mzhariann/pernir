import pandas as pd
import pickle
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm


class CollabVectorized:

    def __init__(self, train_basket_path, similarities_path):
        train_baskets = pd.read_csv(train_basket_path)

        all_baskets = train_baskets[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
        basket_items = all_baskets.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
            .apply(list).reset_index(name='items')

        self.items_by_basket = dict(zip(basket_items['basket_id'], basket_items['items']))

        user_baskets = train_baskets[['basket_id', 'user_id']].drop_duplicates()
        user_baskets = user_baskets.groupby(['user_id'])['basket_id'].apply(list).reset_index(name='baskets')

        self.baskets_by_user = dict(zip(user_baskets['user_id'], user_baskets['baskets']))

        user_sim_dict = {}
        self.user_neighbors = {}

        with open(similarities_path, 'rb') as handle:
            user_sim_dict = pickle.load(handle)

        for key in user_sim_dict:
            if key[0] not in self.user_neighbors:
                self.user_neighbors[key[0]] = []
            self.user_neighbors[key[0]].append(key[1])


    def _baskets_of_user(self, user_id):
        return [self.items_by_basket[basket_id] for basket_id in self.baskets_by_user[user_id]]


    def precompute_for_user(self, user_id, num_items, beta):
        B_u = [self._as_sparse_vector(basket, num_items)
               for basket in self._baskets_of_user(user_id)]
        h_u = self._history_vector(B_u, num_items)
        C_u = self._basket_cooccurrence_matrix(B_u, num_items)

        baskets_of_similar_users = [
            self._baskets_of_user(neighbor)
            for neighbor in self.user_neighbors[user_id]
        ]

        h_N_u, C_N_u = self._history_and_coocc_from_neighbors(h_u, baskets_of_similar_users, num_items)

        h_s_u = beta * h_u + (1 - beta) * h_N_u
        C_s_u = beta * C_u + (1 - beta) * C_N_u

        return h_s_u, C_s_u

    @staticmethod
    def _as_sparse_vector(basket, num_items):
        column_indexes = np.zeros(len(basket))
        row_indexes = basket
        values = np.ones(len(basket))
        return coo_matrix((values, (row_indexes, column_indexes)), shape=(num_items, 1))

    @staticmethod
    def _history_vector(B_u, num_items):
        h_u = coo_matrix(([], ([], [])), shape=(num_items, 1))
        for t in range(len(B_u)):
            h_u += B_u[t] * (1.0 / (len(B_u) - t))
        return h_u

    @staticmethod
    def _basket_cooccurrence_matrix(B_u, num_items):
        C_u = coo_matrix(([], ([], [])), shape=(num_items, num_items))

        for t in range(len(B_u)):

            items = B_u[t].nonzero()[0]
            tmp_row_indexes = []
            tmp_column_indexes = []
            tmp_values = []

            for index_a in range(len(items)):
                item_a = items[index_a]
                for index_b in range(len(items)):
                    item_b = items[index_b]
                    if index_a < index_b:
                        weighted_distance = 1.0 / ((len(B_u) - t) * abs(index_a - index_b))
                        tmp_row_indexes.append(item_a)
                        tmp_column_indexes.append(item_b)
                        tmp_values.append(weighted_distance)
                        # Add symmetric entry as well
                        tmp_row_indexes.append(item_b)
                        tmp_column_indexes.append(item_a)
                        tmp_values.append(weighted_distance)

            C_u += coo_matrix((tmp_values, (tmp_row_indexes, tmp_column_indexes)), shape=(num_items, num_items))

        return C_u


    @staticmethod
    def selection_vector(incomplete_basket, num_items):
        column_indexes = np.zeros(len(incomplete_basket))
        row_indexes = incomplete_basket
        values = [1.0 / (len(incomplete_basket) - i) for i in range(len(incomplete_basket))]

        return coo_matrix((values, (row_indexes, column_indexes)), shape=(num_items, 1))

    @staticmethod
    def _history_and_coocc_from_neighbors(h_u, baskets_of_similar_users, num_items):

        h_N_u = coo_matrix(([], ([], [])), shape=(num_items, 1))
        C_N_u = coo_matrix(([], ([], [])), shape=(num_items, num_items))

        for baskets in baskets_of_similar_users:

            B_v = [CollabVectorized._as_sparse_vector(basket, num_items) for basket in baskets]
            h_v = CollabVectorized._history_vector(B_v, num_items)
            C_v = CollabVectorized._basket_cooccurrence_matrix(B_v, num_items)


            similarity = ((h_u.T * h_v) / (norm(h_u) * norm(h_v))).data[0]

            h_N_u += similarity * h_v
            C_N_u += similarity * C_v

        num_similar_users = len(baskets_of_similar_users)
        h_N_u /= num_similar_users
        C_N_u /= num_similar_users

        return h_N_u, C_N_u
