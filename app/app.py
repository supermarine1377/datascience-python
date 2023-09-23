import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

data = pd.read_csv('dataset/home-data-for-ml-course/train.csv')
feat_name = [
  'LotArea', 
  'YearBuilt', 
  '1stFlrSF', 
  '2ndFlrSF', 
  'FullBath',
  'BedroomAbvGr',
  'TotRmsAbvGrd'
]
X = data[feat_name]
y = data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

print("best_tree_size:", best_tree_size)

final_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=best_tree_size)
final_model.fit(train_X, train_y)
val_predictions_final_model =  final_model.predict(val_X)

val_mae_final_model = mean_absolute_error(val_y, val_predictions_final_model)
print("val_mae_final_model:", val_mae_final_model)
