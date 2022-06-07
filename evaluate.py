import numpy as np
import pandas as pd
import utils

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score


#Individual calculations
def residuals(y,yhat):
    return (y - yhat)

def sse(y,yhat):
    return sum(residuals(y,yhat)**2)

def rmse(y,yhat):
    return sqrt(mean_squared_error(y,yhat))

def ess(y,yhat):
    return sum((yhat-y.mean())**2)

def tss(y,yhat):
    return ess(y,yhat) + sse(y,yhat)


#need more than what they provided
def plot_residuals(x,y,yhat,title='Residual'):
    '''
    Creates a scatterplot showing residual vs independent variable

    Outputs: AxesSubplot (scatterplot)
    Returns: None
    Input:
      (R)     x: independent variable (pd.Series or np.array)
      (R)     y: actual values (pd.Series or np.array)
      (R)  yhat: predicted values (pd.Series or np.array)
      (O) title: title of chart (string).  Default: 'Residual'
    '''
    #get residual
    y=residuals(y,yhat)
    #plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    ax = sns.scatterplot(x=x, y=y,alpha=.05)
    #Format y axis
    if y.max() > 1_000_000: utils.yticks_mm(ax)
    elif y.max() > 2500: utils.yticks_k(ax)
    #Add actual line (y=0)
    plt.axhline(y=0,c='r')
    #Add text
    plt.xlabel('x')
    plt.ylabel('Residual')
    plt.title(title)
    return None

def regression_errors(y,yhat):
    '''
    Takes in actual and predicted values. Returns dataframe of regression performance statistics.
    
    Returns: Pandas DataFrame
    Input:
      (R)    y: actual values (pd.Series or np.array)
      (R) yhat: predicted values (pd.Series or np.array)
    
    '''
    #set index name for dataframe
    if isinstance(yhat,pd.Series): ind=yhat.name
    else: ind='yhat'
    #Create DataFrame with performance stats as columns
    df = pd.DataFrame({
        'sse': [sse(y,yhat)],
        'ess': [ess(y,yhat)],
        'tss': [tss(y,yhat)],
        'mse': [mean_squared_error(y,yhat)],
        'rmse': [rmse(y,yhat)],
        },index=[ind])
    return df

def baseline_mean_errors(y):
    '''
    Takes in actual values. Returns dataframe of regression performance statistics.
    
    Returns: Pandas DataFrame
    Input:
      (R)    y: actual values (pd.Series or np.array)
    '''
    #Create series of yhat_baseline
    if isinstance(y,pd.Series): ind = y.index
    else: ind = range(len(y))
    yhat_b = pd.Series(y.mean(),index=ind)
    #Create DataFrame with performance stats as columns
    df = pd.DataFrame({
        'sse': [sse(y,yhat_b)],
        'mse': [mean_squared_error(y,yhat_b)],
        'rmse': [rmse(y,yhat_b)],
        },index=['yhat_baseline'])
    return df

def better_than_base(y,yhat):
    '''
    Takes in actual and predicted values. Returns True/False on if \
    the model performed better than the dataframe based on rmse.
    
    Returns: Boolean
    Input:
      (R)    y: actual values (pd.Series or np.array)
      (R) yhat: predicted values (pd.Series or np.array)
    
    '''
    #Determine if series or array - use info to create baseline series
    if isinstance(y,pd.Series): ind = y.index
    else: ind = range(len(y))
    yhat_b = pd.Series(y.mean(),index=ind)
    #Get RMSE for model and baseline
    rmse_base = rmse(y, yhat_b)
    rmse_mod = rmse(y,yhat)
    return rmse_mod < rmse_base