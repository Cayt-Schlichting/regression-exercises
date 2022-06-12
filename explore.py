
#Basics
import numpy as np
import pandas as pd
from itertools import combinations, product

#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

#My Modules
import utils

def plot_variable_pairs(df,**kwargs):
    '''
    Creates combinations of numeric columns, then creates a scatterplot and regression line.\
    Do not include encoded columns when calling function.
    
    Outputs: Scatterplot with regression line
    Returns: None
    Inputs: 
     (R)          df: Dataframe containing multiple numeric columns.
     (O) sample_size: number of rows to use when plotting.  Default 50_000
    '''
    #only include numeric datatypes 
    #doesn't currently handle datetimes - would want it to plot that on X
    df = df.select_dtypes(include='number')

    #SCATTERPLOTS
    #pull out sample size
    ss = kwargs.get('sample_size',50_000) # Default 50k
    #If sample size is smaller than df rows, pull out a sample of the data
    if ss < df.shape[0]: df = df.sample(n=ss,random_state=88)

    #get combinations
    combos = combinations(df.columns,2)

    #Loop over combinations and plot
    for pair in combos:
        #Add a chart - lmplot generates facetgrid
        sns.lmplot(data=df,x=pair[0],y=pair[1],line_kws={'color':'red'})
        plt.title(f'{pair[1]} vs {pair[0]}')
        plt.show()
    
    return None

def plot_cat_and_continuous(df,**kwargs):
    '''
    Takes dataframe and plots all categorical variables vs all continuous variables. \
    Subset of categorical columns and continuous columns can be passed.  If not specified,\
    Assumes all objects and boolean columns to be categories and all numeric columns to be continuous
    **DOES NOT HANDLE DATETIMES**
    
    OUTPUTS: Charts
    RETURNS: None
    INPUTS:
      (R)            df: Pandas Dataframe containing categorical and continous columns
      (O)   sample_size: number of rows to use when plotting.  Default 50_000
      (O)      cat_cols: List of categorical columns to be plotted. Default: object and boolean dtypes
      (O)     cont_cols: List of continuous columns to be plotted. Default: numeric dtypes
    '''
    #pull out sample size
    ss = kwargs.get('sample_size',50_000) # Default 50k
    #If sample size is smaller than df rows, pull out a sample of the data
    if ss < df.shape[0]: df = df.sample(n=ss,random_state=88)

    #Get categorical and continuous features
    cats = kwargs.get('cat_cols',df.select_dtypes(include=['bool','object']))
    conts = kwargs.get('cont_cols',df.select_dtypes(include='number'))
    
    #create pairs
    pairs = product(cats,conts)
    
    #Loop over pairs to plot
    for pair in pairs:
        #Cats will be first in the pair
        cat= pair[0]
        cont= pair[1]
        #Plot 3 charts (1x3)
        fig, ax = plt.subplots(1,3,figsize=(12,4),sharey=True)
        fig.suptitle(f'{cont} vs. {cat}')
        
        #First Chart
        plt.subplot(1,3,1)
        sns.boxplot(data=df,x=cat,y=cont)
        #Format y axis
        if df[cont].max() > 1_000_000: utils.yticks_mm(ax[0])
        elif df[cont].max() > 2500: utils.yticks_k(ax[0])
        #Other Charts - shared y axis
        plt.subplot(1,3,2)
        sns.violinplot(data=df,x=cat,y=cont)
        plt.subplot(1,3,3)
        sns.stripplot(data=df,x=cat,y=cont)
        plt.tight_layout()
    
    return None