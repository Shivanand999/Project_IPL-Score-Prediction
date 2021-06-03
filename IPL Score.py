# Importing essential libraries
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Loading the dataset
df = pd.read_csv('e:\\Dell PC\\Downloads\\Online Courses\\Projects\\Project_IPL-First-

Innings-Score-Prediction_DS\\ipl.csv')


df.bat_team.value_counts()


# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)

# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]

# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi 

Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 

'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 

'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 

'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))




###########################################################################

################################



from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


lasso  = Lasso(alpha = 0.01).fit(X_train,y_train)

lasso.score(X_train,y_train)



# Lasso Model using GridSearchCV

lasso  = Lasso()
parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
              
lasso_model = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5).fit(X_train, y_train)

lasso_model.best_params_
lasso_model.best_score_

lasso_model.score(X_train,y_train)

MAE_lasso = mean_absolute_error(lasso_model.predict(X_test), y_test)

lasso_RMSE = np.sqrt(mean_squared_error(lasso_model.predict(X_test), y_test))



###########################################################################

###########################3


Ridge  = Ridge(alpha = 0.4).fit(X_train,y_train)

Ridge.score(X_train,y_train)



# Ridge Model using GridSearchCV

from sklearn.linear_model import Ridge

ridge  = Ridge()
parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}
              
ridge_model = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5).fit(X_train, y_train)

ridge_model.best_params_
ridge_model.best_score_

ridge_model.score(X_train,y_train)

MAE_Ridge = mean_absolute_error(ridge_model.predict(X_test), y_test)

RMSE_ridge = np.sqrt(mean_squared_error(ridge_model.predict(X_test), y_test))






