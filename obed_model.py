#import essential libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#link notebook to drive to access csv file
from google.colab import drive
drive.mount('/content/drive')

#fetch csv file via Pandas Library
house_data = pd.read_csv('/content/drive/MyDrive/Housing.csv')
#delete any empty rows
fil_house_data = house_data.dropna(axis=0)

#check out the read file
fil_house_data.info()

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# fil_house_data['furnishingstatus'] = le.fit_transform(fil_house_data['furnishingstatus'])

#for categorical data change to integer for easy implemetation by sckitLearn
!pip install MultiColumnLabelEncoder
from MultiColumnLabelEncoder import MultiColumnLabelEncoder
Mcle = MultiColumnLabelEncoder()
fil_house_data = Mcle.fit_transform(fil_house_data)

#observe transformation
fil_house_data.head()
fil_house_data.describe()
fil_house_data.columns

#assign the dependent variable
y = fil_house_data.price

#assign a list to other independent variables you desire to predict the behaviour of the dependent variable
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus']

#assign the independent variables to X
X = fil_house_data[features]

#observe
X.head()

#create the model
obed_model = DecisionTreeRegressor(random_state = 1)

#fit the model
obed_model.fit(X,y)

#make predictions
preds = obed_model.predict(X)
print(preds)

#find the MAE
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
