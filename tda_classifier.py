import pickle
with open('tda_weights', 'rb') as f:
    rf = pickle.load(f)


preds = rf.predict(new_X)
