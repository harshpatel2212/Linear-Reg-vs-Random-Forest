# linear Regression Case-Study

import pandas as pd
import numpy as np
import seaborn as sns

cars_data = pd.read_csv("cars_sampled.csv")
cars = cars_data.copy()

# cars.info()
cars.describe()
pd.set_option('display.float_format', lambda x: '%0.3f' % x)
cars.describe()

pd.set_option('display.max_columns', 10)
cars.describe()

# drop unwanted variables
col = np.array(cars.columns)
col_drop = ['dateCrawled','postalCode', 'name','dateCreated','lastSeen']
cars = cars.drop(columns = col_drop, axis = 'column')
print(np.array(cars.columns))

# duplication check
cars.drop_duplicates(keep = 'first',inplace = True)

###############################################################################
# data cleaning
cars.isna().sum()

# variable yearOfRegistration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
# working range - 1950 - 2018

# varible price
price_count = cars['price'].value_counts().sort_index()
sns.boxplot(y= price_count) # conclusion lot of outliers we need to drop them 
sum(cars['price'] > 150000)
sum(cars['price'] < 100)
# working range - 100 to 150000

# variable powerPS
powerPS_count = cars['powerPS'].value_counts().sort_index()
sns.boxplot(y= powerPS_count) # conclusion lot of outliers we need to drop them
sum(cars['powerPS'] > 500)
sum(cars['powerPS'] < 10)
# working range - 10 to 500, here trial and error is supposed to be used and \
# practical experience and information about the property should be considered \
# while taking the decision.

sns.regplot(y = 'price', x = 'powerPS', data = cars, marker = "o", color = 'blue')

# new something to learn op!!.
# common condition way access of data

cars = cars[
        (cars.yearOfRegistration >= 1950) 
      & (cars.yearOfRegistration <= 2018)
      & (cars.price >= 100)
      & (cars.price <= 150000)
      & (cars.powerPS >= 10)
      & (cars.powerPS <= 500)]

# make month in terms of year
cars['monthOfRegistration'] /= 12

# new age variable to have age of the car instead of year and month separately
cars['age'] = (2018 - cars['yearOfRegistration'] + cars['monthOfRegistration'])
cars.age = round(cars.age,2)
cars.age.describe()

# drop yearOfRegistration and monthOfRegistration
cars = cars.drop(['yearOfRegistration','monthOfRegistration'], axis = 1)

###############################################################################
# visual parameters

# age
sns.histplot(cars.age, kde=True)
sns.boxplot(y = cars.age)

# price
sns.histplot(cars.price, kde=True)
sns.boxplot(y = cars.price)

# powerPS
sns.histplot(cars.powerPS, kde=True)
sns.boxplot(y = cars.powerPS)

# regression plot to see the comparison
# age vs price
sns.regplot(x = 'age', y = 'price', data=cars, scatter=True, fit_reg = False)

# powerPS vs price
sns.regplot(x = 'powerPS', y = 'price', data=cars, scatter=True, fit_reg = False)

print(cars.columns)
# checking the relation between the price with all the other variables
# seller
cars.seller.value_counts().sort_index()
pd.crosstab(index = cars.seller, columns = 'count', normalize=True)
sns.countplot(x = 'seller', data = cars)
sns.boxplot(x= 'seller', y = 'price', data=cars)
# private dominants and hence no other categories play role -> insignificant 

# offerType
cars.offerType.value_counts().sort_index()
pd.crosstab(index = cars.offerType, columns = 'count', normalize=True)
sns.countplot(x = 'offerType', data = cars)
sns.boxplot(x= 'offerType', y = 'price', data=cars)
# similar to seller only one category -> insignificant

# abtest
cars.abtest.value_counts().sort_index()
pd.crosstab(index = cars.abtest, columns = 'count', normalize=True)
sns.countplot(x = 'abtest', data = cars)
sns.boxplot(x= 'abtest', y = 'price', data=cars)
# both the category are nearly equivalent no significant influence observed -> insignificant

# vehicleType
cars.vehicleType.value_counts().sort_index() # sorts by name, default is descending.
cars.vehicleType.value_counts()
pd.crosstab(index = cars.vehicleType, columns = 'count', normalize=True)
sns.countplot(y = 'vehicleType', data = cars)
sns.boxplot(y= 'vehicleType', x = 'price', data=cars, )
# considerable dependency keeping in model

# gearbox
cars.gearbox.value_counts().sort_index()
pd.crosstab(index = cars.gearbox, columns = 'count', normalize=True)
sns.countplot(x = 'gearbox', data = cars)
sns.boxplot(x= 'gearbox', y = 'price', data=cars)
# keeping in model

# powerPS
cars.powerPS.value_counts().sort_index()
pd.crosstab(index = cars.powerPS, columns = 'count', normalize=True)
sns.countplot(y = 'powerPS', data = cars)
sns.boxplot(x= 'powerPS', y = 'price', data=cars)
# keeping in the model

# model
cars.model.value_counts()
pd.crosstab(index = cars.model, columns = 'count', normalize=True)
sns.countplot(x = 'model', data = cars)
sns.boxplot(x= 'model', y = 'price', data=cars)
# many categories keeping in the model

# kilometer
cars.kilometer.value_counts()
pd.crosstab(index = cars.kilometer, columns = 'count', normalize=True)
sns.countplot(y = 'kilometer', data = cars)
sns.boxplot(x= 'kilometer', y = 'price', data=cars)
# keeping in model

# fuelType
cars.fuelType.value_counts()
pd.crosstab(index = cars.fuelType, columns = 'count', normalize=True)
sns.countplot(x = 'fuelType', data = cars)
sns.boxplot(x= 'fuelType', y = 'price', data=cars)
# significant kept in the model

# brand
cars.brand.value_counts()
pd.crosstab(index = cars.brand, columns = 'count', normalize=True)
sns.countplot(y = 'brand', data = cars)
sns.boxplot(x= 'brand', y = 'price', data=cars)
# kept in model

# notRepairedDamage
cars.notRepairedDamage.value_counts()
pd.crosstab(index = cars.notRepairedDamage, columns = 'count', normalize=True)
sns.countplot(x = 'notRepairedDamage', data = cars)
sns.boxplot(x= 'notRepairedDamage', y = 'price', data=cars)
# damaged and not repaired have less price as expected -> significant , kept in model

# droping seller, offerType and abTest as they are insignificant
cars = cars.drop(columns=['seller','offerType', 'abtest'], axis = 1)

# To not accidentally loose the analysis.
cars_copy = cars.copy();

# correlation
cars_correlTypes = cars.select_dtypes(exclude = ['object'])
correlation = cars_correlTypes.corr()
round(correlation, 3)
correlation.loc[:,'price'].abs().sort_values(ascending = False)[1:]

###############################################################################
# Model Building and Data Filling.
# -> linear Regression.
# -> Random Forest.

# 1. Omitting missing records.
# 2. Filling the missing data

# omiiting missing data
cars_omit = cars.dropna(axis = 0)

# for model converting categorical variables to dummies
cars_omit = pd.get_dummies(cars_omit,drop_first=True)

###############################################################################
# scikit modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

###############################################################################
# Modelling it into X and y to train the model.

X1 = cars_omit.drop('price',axis = 1)
y1 = cars_omit['price']

# y1 is too large and not so well organozed so we check for log(y1) and getting better result we change y1 to log(y1)
prices = pd.DataFrame({'Before' : y1, 'After' : np.log(y1)})
prices.hist()

# Better y1 = log(y1)
y1 = np.log(y1)

# splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X1,y1, test_size=0.3, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# finding the mean for test data value - base predicted value
base_pred = np.mean(y_test)
print(base_pred)
                       
# repeating it to all over the test data to calculate the RMSE
base_pred = np.repeat(base_pred, len(y_test))

# finding the RMSE value
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

# fitting the model
lgr = LinearRegression()

# model
model_lin1 = lgr.fit(X_train,y_train)

# predicting on test data
cars_predictions_lin1 = lgr.predict(X_test)

# calculating MSE and RMSE
lin_mse1 = mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)

# R squared value - Measure of variablity 
r2_lin_test1 = model_lin1.score(X_test, y_test)
r2_lin_train1 = model_lin1.score(X_train, y_train)
print(r2_lin_test1,r2_lin_train1)

# residual diagnostics
residual1 = y_test - cars_predictions_lin1
sns.regplot(x = cars_predictions_lin1, y = residual1, fit_reg=False)


# =============================================================================
# Random forest 
# =============================================================================

# Model Parameters
rf1 = RandomForestRegressor(max_depth=100, min_samples_leaf= 4, min_samples_split=10,random_state = 1, max_features='auto')

# Model
model_rf1 = rf1.fit(X_train, y_train)

# Prediction on test data set
cars_predictions_rf1 = rf1.predict(X_test)

# Computing MSE and RMSE
rf1_mse = mean_squared_error(y_test, cars_predictions_rf1)
rf1_rmse = np.sqrt(rf1_mse)

# R squared Value
r2_rf1_test = model_rf1.score(X_test, y_test)
r2_rf1_train = model_rf1.score(X_train, y_train)


###############################################################################
# Models with inputed Data
###############################################################################

cars_imputed = cars.apply(lambda x: x.fillna(x.median())\
                          if x.dtype == 'float' else \
                          x.fillna(x.value_counts().index[0]))

cars_imputed.isna().sum()

# for model converting categorical variables to dummies
cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

###############################################################################
# Modelling it into X and y to train the model.

X2 = cars_imputed.drop('price',axis = 1)
y2 = cars_imputed['price']

# y1 is too large and not so well organozed so we check for log(y1) and getting better result we change y1 to log(y1)
prices1 = pd.DataFrame({'Before' : y2, 'After' : np.log(y2)})
prices1.hist()

# Better y1 = log(y1)
y2 = np.log(y2)

# splitting data into train and test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X2,y2, test_size=0.3, random_state=3)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)

# finding the mean for test data value - base predicted value
base_pred1 = np.mean(y_test1)
print(base_pred1)
                       
# repeating it to all over the test data to calculate the RMSE
base_pred1 = np.repeat(base_pred1, len(y_test1))

# finding the RMSE value
base_root_mean_square_error1 = np.sqrt(mean_squared_error(y_test1,base_pred1))
print(base_root_mean_square_error1)

# fitting the model
lgr1 = LinearRegression()

# model
model_lin2 = lgr1.fit(X_train1,y_train1)

# predicting on test data
cars_predictions_lin2 = lgr1.predict(X_test1)

# calculating MSE and RMSE
lin_mse2 = mean_squared_error(y_test1, cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)

# R squared value - Measure of variablity 
r2_lin_test2 = model_lin2.score(X_test1, y_test1)
r2_lin_train2 = model_lin2.score(X_train1, y_train1)
print(r2_lin_test2,r2_lin_train2)

# residual diagnostics
residual2 = y_test1 - cars_predictions_lin2
sns.regplot(x = cars_predictions_lin2, y = residual2, fit_reg=False)


# =============================================================================
# Random forest 
# =============================================================================

# Model Parameters
rf2 = RandomForestRegressor(max_depth=100, min_samples_leaf= 4, min_samples_split=10,random_state = 1, max_features='auto')

# Model
model_rf2 = rf2.fit(X_train1, y_train1)

# Prediction on test data set
cars_predictions_rf2 = rf2.predict(X_test1)

# Computing MSE and RMSE
rf2_mse = mean_squared_error(y_test1, cars_predictions_rf2)
rf2_rmse = np.sqrt(rf2_mse)

# R squared Value
r2_rf2_test = model_rf2.score(X_test1, y_test1)
r2_rf2_train = model_rf2.score(X_train1, y_train1)
print(r2_rf2_test, r2_rf2_train)


# =========================================================================== #
# Conclusion
# =========================================================================== #

print('R Squared Value for training data of Linear Regression omitted data - ',r2_lin_train1)
print('R Squared Value for test data of Linear Regression omitted data - ',r2_lin_test1)
print('R Squared Value for training data of Random Forest omitted data - ',r2_rf1_train)
print('R Squared Value for test data of Random Forest omitted data - ',r2_rf1_test)
print('MSE Value for linear regression of omitted data - ',lin_mse1)
print('RMSE Value for linear regression of omitted data - ',lin_rmse1)
print('MSE Value for Random Forest of omitted data - ',rf1_mse)
print('RMSE Value for Random Forest of omitted data - ',rf1_rmse)
print('R Squared Value for training data of Linear Regression imputed data - ',r2_lin_train2)
print('R Squared Value for test data of Linear Regression imputed data - ',r2_lin_test2)
print('R Squared Value for training data of Random Forest imputed data - ',r2_rf2_train)
print('R Squared Value for test data of Random Forest imputed data - ',r2_rf2_test)
print('MSE Value for linear regression of imputed data - ',lin_mse2)
print('RMSE Value for linear regression of imputed data - ',lin_rmse2)
print('MSE Value for Random Forest of imputed data - ',rf2_mse)
print('RMSE Value for Random Forest of imputed data - ',rf2_rmse)

