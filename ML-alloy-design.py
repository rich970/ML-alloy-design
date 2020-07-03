#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:01:35 2020
TASKS:
    
    # unit cell volume  [AA^3] DONE
    # sns.distplot(a = X['saturation magnetization'], kde = False)  #[Tesla] DONE
    # magnetocrystalline anisotropy constants - 1x3 vector  [K1,K2] [MJ/m3] DONE
    # kind of anisotropy - nan, easy axis, easy plane and easy cone DONE
    # NUmber of binary, ternary, quarternary compisitions etc. DONE
    # Graph of element and number of occurances in alloys  DONE
    # Graph of Hk versus Ms  DONE
    # Do we have any duplicate compositions? DONE
        Find a way to deal with duplicate values - median?
    # Implement feature generature functions
    # Design features based on periodic table inputs:
        build stoichemetry array DONE
        Lp stoichiometry norm (p=1,2,3)
        CW atomic mass DONE
        stoichiometry entropy
        CW valence electrons
        CW group
        CW period DONE
        CW molar volume
        CW melting T DONE
        CW electrical resistivity
        CW electronegativity
        CW enthalpy of formation (Miedema) - total binary combination? DONE
    #Additional features
        unit cell volume
        space group
        
@author: richard
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import alloys as al

plt.close('all')
# Import periodic table data
PT = al.importPT('./data/Periodic-table/periodic_table.xlsx')
# Import Miedema model enthalpies
MM = al.importMM('./data/Miedema-model/Miedema-model-reduced.xlsx')
X = al.importNovamag('./data')

# Number of features:
print('The total number of imported features is {0}'.format(len(X.columns)))
# Fix the 'none' values
X = X.fillna(value=np.nan).copy()
# Find columns with missing values
na_cols = [col for col in X.columns if X[col].isna().any()]
print('number of features with at least one NaN value is {0}'.format(
                                                              len(na_cols)))

# drop colums with mostly NaN values
dropped_cols = []
for col in na_cols:
    count = X[col].isna().sum()
    print('Column \'{0}\' has {1} nan values'.format(col, count))
    if count > 1650:
        print('Dropping colum \'{0}\''.format(col))
        dropped_cols.append(col)
        X = X.drop(col, axis=1).copy()
print('Number of dropped features is: {0}'.format(len(dropped_cols)))

# Select categorical columns
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
print('Number of categorical features is: {0}'.format(len(categorical_cols)))
# Select numerical columns
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
print('Number of numerical features is: {0}'.format(len(numerical_cols)))

# Will count the number of non-zero elements for each column [%]
print('Percentage of values in remaining features which are non-zero:')
print(100*X.notnull().sum()/len(X))

# Anisotropy constant are vectors but the second components is always zero
K = X['magnetocrystalline anisotropy constants'].copy()
for i in range(len(K)):
    try:
        # we have no non zero K2 values, might as well drop them.
        X['magnetocrystalline anisotropy constants'].iloc[i] = K.iloc[i][0]
    except: TypeError # to deal with non-subscriptable 'nan' values

# =============================================================================
# Exlpore the data with some figures and stats
# =============================================================================
# What unique values are there to the 'kind of anisotropy'feature?
print('Types of anisotropy - categorical options:')
print(X['kind of anisotropy'].unique())

# Are all compounds unique?
print('Number of duplicate values:')
print(X['chemical formula'].value_counts())

n_el = al.get_element_occurance(X, PT)

# Plot up the distribution of elements
plt.figure()
n_el = n_el[n_el['count'] > 0]
n_el = n_el.sort_values(['count'], ascending=False)
ch = sns.barplot(x=n_el['element'], y=n_el['count'])
# For reverse order, use n['count'][::-1]
ch.set_xticklabels(ch.get_xticklabels(), rotation=90,
                   horizontalalignment='center')

# Make some sub-datasets grouped by elements
X_Fe = X[X['chemical formula'].str.contains(pat='Fe')]
X_Co = X[X['chemical formula'].str.contains(pat='Co')]
X_Cr = X[X['chemical formula'].str.contains(pat='Cr')]
X_Mn = X[X['chemical formula'].str.contains(pat='Mn')]

plt.figure()
bins = np.arange(0, 2.6, 0.2)
# sns.distplot(a = X['saturation magnetization'], kde = True, label = 'all') 
sns.distplot(a=X_Fe['saturation magnetization'], kde=True, bins=bins, label='Fe')
sns.distplot(a=X_Co['saturation magnetization'], kde=True, bins=bins, label='Co')
sns.distplot(a=X_Cr['saturation magnetization'], kde=True, bins=bins, label='Cr') 
sns.distplot(a=X_Mn['saturation magnetization'], kde=True, bins=bins, label='Mn')
plt.legend()

# Investigate any relationship between K and Ms
X_K = X[X['magnetocrystalline anisotropy constants'].notnull()]
plt.figure()
sns.scatterplot(x=X_K['saturation magnetization'],
                y=abs(X_K['magnetocrystalline anisotropy constants']))

# Calculate correlation statistic
corr = abs(X_K['magnetocrystalline anisotropy constants']).corr(
                X_K['saturation magnetization'], method='spearman')
print('Spearman rank correlation between K and Ms is {0}'.format(corr))

# =============================================================================
# Start building feautures based on the compound's chemical formula
# =============================================================================
stoich_array = al.get_stoich_array(X, PT)
X['Zw'] = al.get_Zw(PT, stoich_array)
X['periodw'] = al.get_Periodw(PT, stoich_array)
X['groupw'] = al.get_Groupw(PT, stoich_array)
X['meltingTw'] = al.get_MeltingTw(PT, stoich_array)
X['miedemaH'] = al.get_Miedemaw(MM, stoich_array)
X['valencew'] = al.get_Valencew(PT, stoich_array)

# Plot the distribution of compisition weighted atomic mass
Z_Fe = 55.8
Z_Co = 58.9
Z_Ni = 58.7
plt.figure()
ch2 = sns.distplot(X['Zw'])
plt.plot([Z_Fe, Z_Fe], [0, ch2.get_ylim()[1]])
plt.plot([Z_Co, Z_Co], [0, ch2.get_ylim()[1]])


# =============================================================================
# Build the Random Forest model
# =============================================================================
my_cols = ['Zw',
           'periodw',
           'groupw',
           'meltingTw',
           'miedemaH',
           'valencew']

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['saturation magnetization'] + my_cols, inplace=True)
y = X['saturation magnetization']
X.drop(['saturation magnetization'], axis=1, inplace=True)


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)
X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()
model = RandomForestRegressor(
    n_estimators=200, random_state=0, max_depth=12)
model.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
# preds = model.predict(X_valid)
# print('MAE:', mean_absolute_error(y_valid, preds))

# Do k-fold cross validation to assess model
folds = 5
cv = ShuffleSplit(n_splits=folds, test_size=0.2, random_state=0)
scores = cross_val_score(
    model, X[my_cols], y, scoring='neg_mean_absolute_error', cv=cv)
print('{0}-fold validation scores: {1}'.format(folds, scores))

preds = model.predict(X[my_cols])
plt.figure()
plt.plot(y, preds, 'o')
plt.xlabel('Saturation magnetisation [T]')
plt.ylabel('Predicted Saturation magnetisation [T]')
