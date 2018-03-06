import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# to fetch data and format it
data = fetch_movielens(min_rating=4.0)

# to print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#to create a model
model = LightFM(loss='warp')

# to train the model
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendations(model, data, user_ids):
    # for the number of users and movies in training the data
    n_users, n_items = data['train'].shape

    # to generate recommendations for each user given as input
    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("      Known_positives :")

        for x in known_positives[:3]:
            print("           %s" % x)

        print("             Recommended:")

        for x in top_items[:3]:
            print("           %s" % x)

        print("")


#sample_recommendations(model, data, [15, 75, 680])

print("Enter any three User IDs - ")

a = input()
b = input()
c = input()

sample_recommendations(model, data, [a, b, c])
