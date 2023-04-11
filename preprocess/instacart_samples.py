import pandas as pd

seed_len = 3
basket_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/valid_baskets.csv'
sample_path = '/Users/mozhdeh/PycharmProjects/psbr/data/instacart_30k/valid_samples.csv'

baskets_df = pd.read_csv(basket_path)

basket_users = baskets_df[['basket_id','user_id']].drop_duplicates()
basket_users_dict = dict(zip(basket_users['basket_id'],basket_users['user_id']))

baskets_df = baskets_df[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()

basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id']\
    .apply(list).reset_index(name='items')
basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['items']))

sample_baskets = []
sample_users = []
sample_input = []
sample_label = []
sample_labels = []
for basket in basket_items_dict:
    items = basket_items_dict[basket]
    for i in range(seed_len,len(items)):
        input_items = items[:i]
        label_item = items[i]
        label_items = items[i:]
        sample_baskets.append(basket)
        sample_users.append(basket_users_dict[basket])
        sample_input.append(input_items)
        sample_label.append(label_item)
        sample_labels.append(label_items)

sample_df = pd.DataFrame(columns=['basket_id','user_id','input_items','label_item', 'label_items'])
sample_df['basket_id'] = sample_baskets
sample_df['user_id'] = sample_users
sample_df['input_items'] = sample_input
sample_df['label_item'] = sample_label
sample_df['label_items'] = sample_labels


print(sample_df.shape)
print(sample_df[['basket_id','user_id']].nunique())
sample_df.to_csv(sample_path, index = False)