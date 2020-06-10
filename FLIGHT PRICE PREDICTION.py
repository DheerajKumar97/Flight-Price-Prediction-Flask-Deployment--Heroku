############# FLIGHT PRICE PREDICTION  ################## 
            
       # Exploratory Data Analysis and Data Preprocessing # 

#########################################################################################################
''' Loading Liraries'''
#########################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pyodbc

pd.set_option('display.max_columns', 500)
#########################################################################################################
''' FETCHING ADTA'''
#########################################################################################################

server = 'DESKTOP-GQDTTAA'
db = 'practice'

conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+db+';Trusted_Connection==yes')

sql ="""
SELECT * FROM DataTrain
"""
df_train = pd.read_sql(sql,conn)
df_train

sql ="""
SELECT * FROM TestSet
"""
df_test = pd.read_sql(sql,conn)
df_test

df_train.shape,df_test.shape
#########################################################################################################
''' DATA PREPROCESSING '''
#########################################################################################################
from pandas_profiling import ProfileReport
FlightProfile_train = ProfileReport(df_train,title="FLIGHT PRICE PREDICTION REPORT",explorative=True)
FlightProfile_test = ProfileReport(df_test,title="FLIGHT PRICE PREDICTION REPORT",explorative=True)

df_train.count()
df_test.count()


df_train.dtypes
df_test.dtypes

fg = df_train.append(df_test, sort = False)

fg.isna().sum().sum()
fg.isna().sum()
#########################################################################################################
''' FEATURE ENGINEERING '''
#########################################################################################################

fg['Date'] = fg['Date_of_Journey'].str.split('/').str[0]
fg['Date'] = fg['Date'].astype(int)

fg['Month'] = fg['Date_of_Journey'].str.split('/').str[1]
fg['Month']=fg['Month'].astype(int)

fg['Year'] = fg['Date_of_Journey'].str.split('/').str[2]
fg['Year'] =fg['Year'].astype(int)

fg = fg.drop(['Date_of_Journey'],axis = 1)

fg.count() 

fg.dtypes

fg['Arrival_Time'] =fg['Arrival_Time'].str.split(' ').str[0]
fg['Arrival_Hour'] = fg['Arrival_Time'].str.split(':').str[1]

fg['Arrival_Hour'] = fg['Arrival_Time'].str.split(':').str[1]
fg['Arrival_Hour'] = fg['Date'].astype(int)

fg['Arrival_Minute'] = fg['Arrival_Time'].str.split(':').str[2]
fg['Arrival_Minute'] = fg['Date'].astype(int)

fg = fg.drop(['Arrival_Time'],axis=1)

fg['Total_Stops'].isna().sum()

fg[fg['Total_Stops'].isna() == True]

fg['Total_Stops'] = fg['Total_Stops'].fillna('1 stop')

fg['Total_Stops'].isna().sum()

fg['Total_Stops'] = fg['Total_Stops'].replace('non-stop','0 stop')

fg['Total_Stops'] = fg['Total_Stops'].str.split(' ').str[0]

fg['Total_Stops'] =fg['Total_Stops'].astype(int)

fg['Departure_Minute'] = fg['Dep_Time'].str.split(':').str[1]

fg['Departure_Hour'] = fg['Dep_Time'].str.split(':').str[0]

fg['Departure_Hour'] = fg['Departure_Hour'].astype(int)

fg['Departure_Minute'] = fg['Departure_Minute'].astype(int)

fg = fg.drop(['Dep_Time'],axis =1)

fg['Route_1']= fg['Route'].str.split('→ ').str[0]

fg['Route_2']= fg['Route'].str.split('→ ').str[1]

fg['Route_3']= fg['Route'].str.split('→ ').str[2]

fg['Route_4']= fg['Route'].str.split('→ ').str[3]

fg['Route_5']= fg['Route'].str.split('→ ').str[4]

fg['Price'].fillna((fg['Price'].mean()),inplace=True)

fg['Route_1'].fillna("None",inplace=True)

fg['Route_2'].fillna("None",inplace=True)

fg['Route_3'].fillna("None",inplace=True)

fg['Route_4'].fillna("None",inplace=True)

fg = fg.drop(['Route'],axis=1)

fg = fg.drop(['Duration'],axis=1)

##########################################
''' [ Encoding Categorical Value ]'''
##########################################
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

fg["Airline"]=encoder.fit_transform(fg['Airline'])

fg["Source"]=encoder.fit_transform(fg['Source'])

fg["Destination"]=encoder.fit_transform(fg['Destination'])

fg["Additional_Info"]=encoder.fit_transform(fg['Additional_Info'])

fg["Route_1"]=encoder.fit_transform(fg['Route_1'])

fg["Route_2"]=encoder.fit_transform(fg['Route_2'])

fg["Route_3"]=encoder.fit_transform(fg['Route_3'])

fg["Route_4"]=encoder.fit_transform(fg['Route_4'])

fg["Route_5"] = fg["Route_5"].astype(str)

fg["Route_5"]=encoder.fit_transform(fg['Route_5'])

############################################
''' Outliers Detection and removal '''
############################################
def outlier(x):
    high=0
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    iqr = q3-q1
    low = q1-1.5*iqr
    high += q3+1.5*iqr
    outlier = (x.loc[(x < low) | (x > high)])
    return(outlier)

q1 =fg['Price'].quantile(.25)
q3 = fg['Price'].quantile(.75)
iqr = q3-q1

fg = fg[~((fg['Price'] < (q1 - 1.5 *iqr))  |  (fg['Customer_Lifetime_Value'] > (q3+ 1.5 * iqr)))]
print(fg)

''' Data Visualization '''

sns.boxplot(x='Airline',y='Price',hue='Source',data=fg)

sns.lmplot(x='Airline',y='Price',hue='Source',data=fg)

sns.regplot(x=fg['Airline'], y=fg['Price'], fit_reg=False)

sns.lineplot(x='Date',y='Price',data=fg)

sns.lineplot(x='Total_Stops',y='Arrival_Hour',data=fg)

sns.pairplot(fg,hue='Source')

#########################################################################################################
''' FEATURE SELECTION '''
#########################################################################################################

fg = fg.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)

#########################################################################################################
''' Seperate Dependent and Indepedent Variables '''
#########################################################################################################

x=fg.drop(['Price'],axis=1)

y=fg.Price

#########################################################################################################
''' Train Test Split '''
#########################################################################################################

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

#########################################################################################################
''' Feature selection '''
#########################################################################################################

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

model=SelectFromModel(Lasso(alpha=0.005,random_state=0))

model.fit(x_train,y_train)

model.get_support()

selected_features=x_train.columns[(model.get_support())]

x_train=x_train.drop(['Year'],axis=1)

x_test=x_test.drop(['Year'],axis=1)

x_train=x_train.drop(['Arrival_Minute'],axis=1)

x_test=x_test.drop(['Arrival_Minute'],axis=1)
#########################################################################################################
''' Model Building '''
#########################################################################################################

############################################
''' STATSMODEL OLS  '''
############################################

import statsmodels.api as sm
model =sm.OLS(y_train,x_train).fit()

model.summary()

#############################################
''' LINEAR REGRESSION '''
#############################################

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

lr_pred = lr.predict(x_tets)

from sklearn import metrics

lr_RMSE = np.sqrt(metrics.mean_squared_error(y_test,lr_pred))

from sklearn.metrics import r2_score

r2_score(y_test, lr_pred)

''' PICKLE MODEL '''
import pickle

pickle.dump(lr,open('lr_model.pkl','wb'))

##################################################
''' RANDOM FOREST '''
##################################################
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100,random_state=42)

reg=regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn import metrics

rf_RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

from sklearn.metrics import r2_score

rf_r2_score = r2_score(y_test,y_pred)

''' PICKLE MODEL '''
import pickle

pickle.dump(regressor,open('rf_model.pkl','wb'))

#####################################################
''' RANDOM FOREST CROSS VALIDATION '''
#######################################################
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(x_train,y_train)

y_predRF=rf_random.predict(x_test)

random_best= rf_random.best_estimator_.predict(x_train)

errors1 = abs(random_best - y_train)

mape = np.mean(100 * (errors1 / y_train))

rf_Score = 100 - mape

print(rf_Score)

####################################################
''' XGBOOST '''
####################################################

from xgboost import XGBRegressor

xgb = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.8, learning_rate = 0.5, max_depth = 6, 
                          alpha = 10, n_estimators = 1000)

xgb.fit(x_train,y_train)

preds = xgb.predict(x_test)

from sklearn import metrics

print('Root Mean SQUARED ERROR   =  ',np.sqrt(metrics.mean_squared_error(y_test,preds)))

from sklearn.metrics import r2_score

r2_score(y_test, preds)

#####################################################
''' XGBOOST CROSS VALIDATION '''
#####################################################

from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [7],
    'criterion' :['gini', 'entropy']
}

CV_XGB1 = GridSearchCV(estimator=xgb, param_grid=param_grid, cv= 10)

CV_XGB1.fit(x_train, y_train)

XGB1=XGBRegressor(random_state=40, max_features='auto', n_estimators= 1300, max_depth=15, criterion='gini')

XGB1_preds = XGB1.predict(x_test)

from sklearn import metrics

print('Root Mean SQUARED ERROR   =  ',np.sqrt(metrics.mean_squared_error(y_test,XGB1_preds)))

from sklearn.metrics import r2_score

r2_score(y_test, preds)

#############################################################################################################
