import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

house_data = pd.read_csv('/content/drive/MyDrive/Housing.csv')
fil_house_data = house_data.dropna(axis=0)

fil_house_data.info()

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# fil_house_data['furnishingstatus'] = le.fit_transform(fil_house_data['furnishingstatus'])

!pip install MultiColumnLabelEncoder
from MultiColumnLabelEncoder import MultiColumnLabelEncoder
Mcle = MultiColumnLabelEncoder()
fil_house_data = Mcle.fit_transform(fil_house_data)

fil_house_data.head()
fil_house_data.describe()
fil_house_data.columns

y = fil_house_data.price

features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus']

X = fil_house_data[features]

X.head()

obed_model = DecisionTreeRegressor(random_state = 1)

obed_model.fit(X,y)

preds = obed_model.predict(X)
print(preds)

mean_absolute_error(y, preds)

# import export_graphviz
from sklearn.tree import export_graphviz

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
dot_data = export_graphviz(obed_model, out_file = None,
               feature_names =['area', 'bedrooms', 'bathrooms', 'stories'])

import pydotplus
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
