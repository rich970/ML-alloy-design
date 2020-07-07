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
        Compound radix
        Lp stoichiometry norm (p=1,2,3)
        CW atomic mass DONE
        stoichiometry entropy
        CW valence electrons DONE
        CW group DONE
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
import pandas as pd
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

# Plot up the distribution of elements
n_el = al.get_element_occurance(X, PT)
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
sns.distplot(a = X['saturation magnetization'], kde = True, label = 'all') 
sns.distplot(a=X_Fe['saturation magnetization'], kde=True, bins=bins, label='Fe')
sns.distplot(a=X_Co['saturation magnetization'], kde=True, bins=bins, label='Co')
sns.distplot(a=X_Cr['saturation magnetization'], kde=True, bins=bins, label='Cr') 
sns.distplot(a=X_Mn['saturation magnetization'], kde=True, bins=bins, label='Mn')
plt.legend()

# Investigate any relationship between K and Ms
# X_K = X[X['magnetocrystalline anisotropy constants'].notnull()]
# plt.figure()
# sns.scatterplot(x=X_K['saturation magnetization'],
#                 y=abs(X_K['magnetocrystalline anisotropy constants']))

# Calculate correlation statistic
# corr = abs(X_K['magnetocrystalline anisotropy constants']).corr(
#                 X_K['saturation magnetization'], method='spearman')
# print('Spearman rank correlation between K and Ms is {0}'.format(corr))

# =============================================================================
# Start building feautures based on the compound's chemical formula
# =============================================================================
stoich_array = al.get_stoich_array(X, PT)
X['stoicentw'] = al.get_StoicEntw(stoich_array)
# X['Zw'] = al.get_Zw(PT, stoich_array)
X['compoundradix'] = al.get_CompoundRadix(PT, X)
X['periodw'] = al.get_Periodw(PT, stoich_array)
X['groupw'] = al.get_Groupw(PT, stoich_array)
X['meltingTw'] = al.get_MeltingTw(PT, stoich_array)
X['miedemaH'] = al.get_Miedemaw(MM, stoich_array)
X['valencew'] = al.get_Valencew(PT, stoich_array)

# # How many binary, ternary, quaternary compounds do we have?
# total_compound_radix = compoundradix.value_counts()
# print('We have {0} binary compounds'.format(total_compound_radix[2]))
# print('We have {0} ternary compounds'.format(total_compound_radix[3]))
# print('We have {0} quarternary compounds'.format(total_compound_radix[4]))

# Plot the distribution of compisition stoicheometry entropy
plt.figure()
ch2 = sns.distplot(X['stoicentw'])

# =============================================================================
# Build the Random Forest model
# =============================================================================

my_cols = ['stoicentw',
           'compoundradix',
           'periodw',
           'groupw',
           'meltingTw',
           'valencew',
           'miedemaH']

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['saturation magnetization'] + my_cols, inplace=True)

# Round the saturation magnetization to 1.d.p
X['saturation magnetization'] = pd.to_numeric(
    X['saturation magnetization']).round(decimals=2)

# Drop alloys which are below magnetic cutoff
X.drop(X[X['saturation magnetization'] < 0.18].index, axis=0, inplace=True)

# Group duplicates by chemical formula and replace values with median
X = X.groupby(by='chemical formula').median()
X.index = range(len(X))

y = X['saturation magnetization']
# X.drop(['saturation magnetization'], axis=1, inplace=True)

# Evaluate the effect of magnetisation cutoff:
# cs = pd.DataFrame(columns=['n_estimators', 'score'])
# for n_estimators in np.arange(10, 500, 10):

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)
X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()

model = RandomForestRegressor(
    n_estimators=270, random_state=0, max_depth=16)
model.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
# preds = model.predict(X_valid)
# print('{0}: MAE: {1}'.format(i, mean_absolute_error(y_valid, preds)))

# Do k-fold cross validation to assess model
folds = 5
cv = ShuffleSplit(n_splits=folds, test_size=0.2, random_state=0)
scores = cross_val_score(
    model, X[my_cols], y, scoring='neg_mean_absolute_error', cv=cv)
print('------------------------------------')
print('Mean {0}-fold validation score: {1:.3f}'.format(folds,
                                                       abs(scores).mean()))
print('------------------------------------')
# cs = cs.append({'depth': n_estimators, 'score': abs(scores).mean()}, ignore_index=True)

preds = model.predict(X[my_cols])
plt.figure()
plt.plot(y, preds, 'o')
plt.xlabel('Saturation magnetisation [T]')
plt.ylabel('Predicted Saturation magnetisation [T]')

# =============================================================================
# Test with literature alloy data
# =============================================================================

# Ni-Mn shows a peak around 10% Mn, so not a monotomic relationship.
if False:
    X_NiMn = pd.DataFrame(['Ni40Mn1', 'Ni40Mn2', 'Ni40Mn3', 'Ni40Mn4', 'Ni40Mn5',
                           'Ni40Mn6', 'Ni40Mn7', 'Ni40Mn8', 'Ni40Mn9'],
                          columns=['chemical formula'])
    stoich_array_NiMn = al.get_stoich_array(X_NiMn, PT)
    X_NiMn['stoicentw'] = al.get_StoicEntw(stoich_array_NiMn)
    X_NiMn['Zw'] = al.get_Zw(PT, stoich_array_NiMn)
    X_NiMn['compoundradix'] = al.get_CompoundRadix(PT, X_NiMn)
    X_NiMn['periodw'] = al.get_Periodw(PT, stoich_array_NiMn)
    X_NiMn['groupw'] = al.get_Groupw(PT, stoich_array_NiMn)
    X_NiMn['meltingTw'] = al.get_MeltingTw(PT, stoich_array_NiMn)
    X_NiMn['miedemaH'] = al.get_Miedemaw(MM, stoich_array_NiMn)
    X_NiMn['valencew'] = al.get_Valencew(PT, stoich_array_NiMn)

    preds_NiMn = model.predict(X_NiMn[my_cols])

    # Calculate atomic fractions for NiMn compositions
    at_fraction = pd.DataFrame()
    for i in range(len(stoich_array_NiMn)):
        compound = stoich_array_NiMn.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()
        print(cols)
        at_fraction = at_fraction.append(
            compound.iloc[cols]/sum(compound.iloc[cols]))

    # Convert Tesla to emu/g
    rho_Ni = 8.9  # [g/cm3]
    rho_Mn = 7.4  # [g/cm3]
    rho_NiMn = (rho_Ni*at_fraction['Ni'] + rho_Mn*at_fraction['Mn'])
    preds_NiMn = preds_NiMn/((4*np.pi*1e-4)*rho_NiMn)

    # Convert bohr magnetron to emu/g
    Exp_NiMn = pd.Series(data=[0.6, 0.76, 0.78, 0.79, 0.81, 0.8, 0.72],
                         index=[0, 0.04, 0.055, 0.07, 0.1, 0.13, 0.165])
    M_Ni = 58.693
    M_Mn = 54.938
    formula_mass = ((M_Ni*(1-Exp_NiMn.index) + M_Mn*Exp_NiMn.index))/6.022e23
    Exp_NiMn = (Exp_NiMn*9.27e-21)/formula_mass

    plt.figure()
    sns.scatterplot(x=at_fraction['Mn'], y=preds_NiMn)
    sns.scatterplot(x=Exp_NiMn.index, y=Exp_NiMn.array)
    plt.xlabel('Mn concentration [at.%]')
    plt.ylabel('Magnetisation [emu/g]')

# Fe-Mn.
if True:
    X_FeMn = pd.DataFrame(['Fe10Mn1', 'Fe10Mn2', 'Fe10Mn3', 'Fe10Mn4', 'Fe10Mn5',
                           'Fe10Mn6', 'Fe10Mn7', 'Fe10Mn8', 'Fe10Mn9'],
                          columns=['chemical formula'])
    stoich_array_FeMn = al.get_stoich_array(X_FeMn, PT)
    X_FeMn['stoicentw'] = al.get_StoicEntw(stoich_array_FeMn)
    X_FeMn['Zw'] = al.get_Zw(PT, stoich_array_FeMn)
    X_FeMn['compoundradix'] = al.get_CompoundRadix(PT, X_FeMn)
    X_FeMn['periodw'] = al.get_Periodw(PT, stoich_array_FeMn)
    X_FeMn['groupw'] = al.get_Groupw(PT, stoich_array_FeMn)
    X_FeMn['meltingTw'] = al.get_MeltingTw(PT, stoich_array_FeMn)
    X_FeMn['miedemaH'] = al.get_Miedemaw(MM, stoich_array_FeMn)
    X_FeMn['valencew'] = al.get_Valencew(PT, stoich_array_FeMn)

    preds_FeMn = model.predict(X_FeMn[my_cols])

    # Calculate atomic fractions for NiMn compositions
    at_fraction = pd.DataFrame()
    for i in range(len(stoich_array_FeMn)):
        compound = stoich_array_FeMn.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()
        print(cols)
        at_fraction = at_fraction.append(
            compound.iloc[cols]/sum(compound.iloc[cols]))

    # Convert Tesla to emu/g
    rho_Fe = 7.87  # [g/cm3]
    rho_Mn = 7.43  # [g/cm3]
    rho_FeMn = (rho_Fe*at_fraction['Fe'] + rho_Mn*at_fraction['Mn'])
    preds_FeMn = preds_FeMn/((4*np.pi*1e-4)*rho_FeMn)

    # Convert bohr magnetron to emu/g
    Exp_FeMn = pd.Series(data=[1.40, 1.51, 1.63, 1.76, 1.88, 2.00, 2.10, 2.34],
                         index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    M_Fe = 55.845
    M_Mn = 54.938
    formula_mass = ((M_Fe*(1-Exp_FeMn.index) + M_Mn*Exp_FeMn.index))/6.022e23
    Exp_FeMn = (Exp_FeMn*9.27e-21)/formula_mass

    plt.figure()
    sns.scatterplot(x=at_fraction['Fe'], y=preds_FeMn)
    sns.scatterplot(x=Exp_FeMn.index, y=Exp_FeMn.array)
    plt.xlabel('Fe concentration [at.%]')
    plt.ylabel('Magnetisation [emu/g]')

# Fe-Co.  NOT CORRECT EXP DATA YET!
if False:
    X_FeCo = pd.DataFrame(['Fe10Co1', 'Fe10Co2', 'Fe10Co3', 'Fe10Co4', 'Fe10Co5',
                           'Fe10Co6', 'Fe10Co7', 'Fe10Co8', 'Fe10Co9'],
                          columns=['chemical formula'])
    stoich_array_FeCo = al.get_stoich_array(X_FeCo, PT)
    X_FeCo['stoicentw'] = al.get_StoicEntw(stoich_array_FeCo)
    X_FeCo['Zw'] = al.get_Zw(PT, stoich_array_FeCo)
    X_FeCo['compoundradix'] = al.get_CompoundRadix(PT, X_FeCo)
    X_FeCo['periodw'] = al.get_Periodw(PT, stoich_array_FeCo)
    X_FeCo['groupw'] = al.get_Groupw(PT, stoich_array_FeCo)
    X_FeCo['meltingTw'] = al.get_MeltingTw(PT, stoich_array_FeCo)
    X_FeCo['miedemaH'] = al.get_Miedemaw(MM, stoich_array_FeCo)
    X_FeCo['valencew'] = al.get_Valencew(PT, stoich_array_FeCo)

    preds_FeCo = model.predict(X_FeCo[my_cols])

    # Calculate atomic fractions for FeCo compositions
    at_fraction = pd.DataFrame()
    for i in range(len(stoich_array_FeCo)):
        compound = stoich_array_FeCo.iloc[i]  # take slice for each compound
        cols = compound.to_numpy().nonzero()
        print(cols)
        at_fraction = at_fraction.append(
            compound.iloc[cols]/sum(compound.iloc[cols]))

    # Convert Tesla to emu/g
    rho_Fe = 7.87  # [g/cm3]
    rho_Co = 8.86  # [g/cm3]
    rho_FeCo = (rho_Fe*at_fraction['Fe'] + rho_Co*at_fraction['Co'])
    preds_FeCo = preds_FeCo/((4*np.pi*1e-4)*rho_FeCo)

    # Convert bohr magnetron to emu/g
    Exp_FeCo = pd.Series(data=[1.40, 1.51, 1.63, 1.76, 1.88, 2.00, 2.10, 2.34],
                         index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    M_Fe = 55.845
    M_Co = 58.933
    formula_mass = ((M_Fe*(1-Exp_FeCo.index) + M_Co*Exp_FeCo.index))/6.022e23
    Exp_FeCo = (Exp_FeCo*9.27e-21)/formula_mass

    plt.figure()
    sns.scatterplot(x=at_fraction['Fe'], y=preds_FeCo)
    sns.scatterplot(x=Exp_FeCo.index, y=Exp_FeCo.array)
    plt.xlabel('Fe concentration [at.%]')
    plt.ylabel('Magnetisation [emu/g]')
