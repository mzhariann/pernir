import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pickle
import time
import numpy as np

from models.collab_vectorized import CollabVectorized


def run_one(model, test_samples, k, latency_log, size_log):
    seed = 42

    num_items = 49689

    alpha = 0.3
    beta = 0.3

    num_users_for_benchmark = 20
    num_incomplete_baskets_for_user = 5

    np.random.seed(seed)

    # Choose a random set of users
    users = np.random.choice(sorted(test_samples.user_id.unique()),
                             size=num_users_for_benchmark, replace=False)

    durations = []

    for user in users:
        # Precompute static parts of model for users, not measured, can happen offline
        h_s_u, C_s_u = model.precompute_for_user(user, num_items, beta)
        # Choose physical representation for the matrices
        h_s_u_opt = h_s_u.tocsc(copy=True)
        C_s_u_opt = C_s_u.tocsc(copy=True)

        size_log.write(f"model_size\t{user}\t{k}\t{h_s_u.count_nonzero()}\t{C_s_u.count_nonzero()}\n")

        all_incomplete_baskets = test_samples[test_samples.user_id == user].input_items.apply(eval).tolist()

        if len(all_incomplete_baskets) > 1:
            # Choose a random set of incomplete baskets as query inputs
            incomplete_baskets = np.random.choice(all_incomplete_baskets, num_incomplete_baskets_for_user)

            for incomplete_basket in incomplete_baskets:
                # Query starts
                start_time = time.time()

                # Compute selection vector from incomplete basket
                f_c = model.selection_vector(incomplete_basket, num_items).tocsc(copy=True)
                # Compute scores for all potential items to recommend
                scores = alpha * h_s_u_opt + (1 - alpha) * C_s_u_opt * f_c

                # Query ends
                duration = time.time() - start_time

                # Log duration
                print(f"latency\t{k}\t{user}\t{len(incomplete_basket)}\t{duration * 1000}")
                latency_log.write(f"latency\t{k}\t{user}\t{len(incomplete_basket)}\t{duration * 1000}\n")
                durations.append(duration)

    print(k, 'median', np.median(durations) * 1000, 'p90', np.percentile(durations, 90) * 1000)


test_sample_path = 'data/instacart_30k/test_samples.csv'
test_samples = pd.read_csv(test_sample_path)

train_basket_path = 'data/instacart_30k/train_baskets.csv'

collab10 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_10.pickle')
collab20 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_20.pickle')
collab50 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_50.pickle')
collab100 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_100.pickle')
collab200 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_200.pickle')
collab250 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_250.pickle')
collab300 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_300.pickle')
collab400 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_400.pickle')
collab500 = CollabVectorized(train_basket_path, 'data/instacart_30k/user_sim_500.pickle')

with open('latencies.csv', 'w') as latency_log:
    with open('sizes.csv', 'w') as size_log:

        run_one(collab10, test_samples, 10, latency_log, size_log)
        run_one(collab20, test_samples, 20, latency_log, size_log)
        run_one(collab50, test_samples, 50, latency_log, size_log)
        run_one(collab100, test_samples, 100, latency_log, size_log)
        run_one(collab200, test_samples, 200, latency_log, size_log)
        run_one(collab250, test_samples, 250, latency_log, size_log)
        run_one(collab300, test_samples, 300, latency_log, size_log)
        run_one(collab400, test_samples, 400, latency_log, size_log)
        run_one(collab500, test_samples, 500, latency_log, size_log)

