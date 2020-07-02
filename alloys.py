#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:15:09 2020
alloys class.
This collection of functions is used to import data from the Novamag database
generate features for the machine learning algorithms.

@author: richard
"""

import pandas as pd
import os
import numpy as np
from itertools import combinations

def importNovamag(root_dir):
    """
    Auto import all json files in the Novamag database, creating a 
    pandas dataframe object.
    Parameters
    ----------
    root_dir : string
        location of root directory where all the Novamag data is stored.

    Returns
    -------
    X : dataframe
        Output dataframe of the Novamag database.

    """
    
    X = pd.DataFrame()
    ls = dict()
    failedfiles = []
    
    for dirName, subdirList, fileList in os.walk(root_dir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname.endswith('.json'):
               filepath = dirName + '/' + fname
               try:
                   df = pd.read_json(filepath,encoding='Latin')
                   ls = {**df.properties.chemistry, **df.properties.crystal,
                         **df.properties.magnetics}
                   X = X.append(pd.DataFrame.from_dict(ls)) 
               
               except ValueError:
                   print('Import failed')
                   failedfiles.append(filepath)
    X.index = range(len(X)) #Fix the index
    return X

def get_K_mag(X):
    K = X['magnetocrystalline anisotropy constants'].copy()
    for i in range(len(K)):
        try: 
            # Turns out we have no non-zero K2 values, so the magnitude is
            # just the K1 value.
            X['magnetocrystalline anisotropy constants'].iloc[i] = K.iloc[i][0]
        except: TypeError  #to deal with 'nan' values which are vectors
    return X

def get_element_occurance(X, PT):
    formulas = X['chemical formula'].copy()
    
    #Get a list of element symbols and sort in order of descending string length
    symbols = PT['symbol'].copy()
    s = symbols.str.len().sort_values(ascending = False).index
    symbols = symbols.reindex(s)

    #Calculate the occurance of each elements
    n_el = pd.DataFrame(columns = ['element', 'count'])
    for el in symbols:
        regex_list = formulas.str.extractall(pat = 
                                             "(?P<element>{0})(?P<digit>\d*)".format(el))
        regex_list = regex_list.droplevel(level=1).copy() #drop the multi-indexing that
                                                          #'extractall' creates
        count = len(regex_list)
        n_el=n_el.append({'element':el,'count': count}, ignore_index = True)
        print('Number of compounds containing {0} is {1}'.format(el,count))
        
        #Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
                                          to_replace = regex_list.element
                                          +regex_list.digit
                                          ,value ='', regex = True)
    return n_el

def get_compound_radix(X, PT):
    formulas = X['chemical formula'].copy()
    X['compound_radix']=np.zeros(len(X)) #Make a new column for the
                                         #compound index i.e. 2 = binary
    
    #Get a list of symbols and sort in order of descending string length
    symbols = PT['symbol'].copy()
    s = symbols.str.len().sort_values(ascending = False).index
    symbols = symbols.reindex(s)

    for el in symbols:
        regex_list = formulas.str.extractall(pat = 
                                             "(?P<element>{0})(?P<digit>\d*)".format(el))
        regex_list = regex_list.droplevel(level=1).copy() #drop the multi-indexing that
                                                          #'extractall' creates
        
        #Remove the elements we have just found from the formulas list
        formulas[regex_list.index] = formulas[regex_list.index].replace(
                                          to_replace = regex_list.element
                                          +regex_list.digit
                                          ,value ='', regex = True)
        
        #Use the regex indices to update the compound radix column
        X['compound_radix'][regex_list.index] += 1
        
    #How many binary, ternary, quaternary compounds do we have?
    total_compound_radix = X['compound_radix'].value_counts()
    print('We have {0} binary compounds'.format(total_compound_radix[2]))
    print('We have {0} ternary compounds'.format(total_compound_radix[3]))
    print('We have {0} quarternary compounds'.format(total_compound_radix[4]))
    return X

def get_stoich_array(X, PT):
    if type(X) == pd.DataFrame:
        formulas = X['chemical formula'].copy()  #if user passes whole of Novamag
    else:
        formulas = pd.Series(X) #if user passes a single chemical formula string
        print(formulas)
    #Get a list of element symbols and sort in order of descending string length
    #Need longest first as elements like S will be picked up within Si, As, Sb etc.
    symbols = PT['symbol'].copy()  
    s = symbols.str.len().sort_values(ascending = False).index
    symbols = symbols.reindex(s)
     
    #Will encode chemical formula data in a large array
    stoich_array = pd.DataFrame(np.zeros([len(formulas),len(symbols)])) 
    stoich_array.columns = symbols.copy()
    stoich_array.index = formulas.copy()

    for el in symbols:
        regex_list = formulas.str.extractall(pat = 
                                             "(?P<element>{0})(?P<digit>\d*)".format(el))
        regex_list = regex_list.droplevel(level=1).copy() #drop the multi-indexing that
                                                          #'extractall' creates
        count = len(regex_list)
        if count>0: 
            #add the number of atoms to the correct element column in the stoich array
            stoich_array[el][regex_list.index] = regex_list.digit

        #Remove the elements we have just found from the formulas list
        formulas[regex_list.index] =formulas[regex_list.index].replace(
            to_replace=regex_list.element+regex_list.digit,
            value ='', regex = True)

    #Need to rewrite the string numbers as integers in our stoichiometry array
    stoich_array = stoich_array.fillna(0)
    for col in stoich_array.columns:
        stoich_array[col] = stoich_array[col].astype(int)
        
    return stoich_array

def get_Zw(PT, stoich_array):
    Zw=pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
            compound = stoich_array.iloc[i] #take slice for each compound
            cols = compound.to_numpy().nonzero()     #nonzero elements columns
            at_fraction = compound.iloc[cols]/sum(compound.iloc[cols])
            Zw.iloc[i] = np.dot(at_fraction,
                                      PT.loc[compound.index[cols]]['atomic_weight'])
    return Zw

def get_Periodw(PT, stoich_array):
    periodw=pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
            compound = stoich_array.iloc[i] #take slice for each compound
            cols = compound.to_numpy().nonzero()     #nonzero elements columns 
            at_fraction = compound.iloc[cols]/sum(compound.iloc[cols])
            periodw.iloc[i] = np.dot(at_fraction,
                                      PT.loc[compound.index[cols]]['period'])
    return periodw

def get_MeltingTw(PT, stoich_array):
    meltingTw=pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
            compound = stoich_array.iloc[i] #take slice for each compound
            cols = compound.to_numpy().nonzero()     #nonzero elements columns 
            at_fraction = compound.iloc[cols]/sum(compound.iloc[cols])
            meltingTw.iloc[i] = np.dot(at_fraction,
                                      PT.loc[compound.index[cols]]['melting_point'])
    return meltingTw

def get_Valencew(PT, stoich_array):
    valencew=pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
            compound = stoich_array.iloc[i] #take slice for each compound
            cols = compound.to_numpy().nonzero()     #nonzero elements columns 
            at_fraction = compound.iloc[cols]/sum(compound.iloc[cols])
            valencew.iloc[i] = np.dot(at_fraction,
                                      PT.loc[compound.index[cols]]['valence'])
    return valencew

def get_Miedemaw(MM, stoich_array):
    miedemaw=pd.Series(np.zeros(len(stoich_array)))
    for i in range(len(stoich_array)):
            compound = stoich_array.iloc[i] #take slice for each compound
            cols = compound.to_numpy().nonzero() 
            at_fraction = compound.iloc[cols]/sum(compound.iloc[cols])
            #Calculate Miedema enthalpy by summing all binary combinations of elements
            comb = combinations(at_fraction.index, 2)
            for el in list(comb):
                try:
                  miedemaw.iloc[i]+=4*at_fraction[el[0]] \
                      *at_fraction[el[1]] \
                      *MM.loc[el[0]][el[1]]
                
                except KeyError:
                  miedemaw.iloc[i] = np.nan  
          
    return miedemaw


