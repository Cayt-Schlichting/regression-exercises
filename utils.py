import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

###### GRAPHING #######
#IMPROVEMENTS TO MAKE: 
# 1) determine type of plot (facetgrid or axes subplot), then set appropriately
# 2) consider making an x vs y parameters
# 3) consider making millions/thousands parameter
# 4) create an alpha selector - for opacity based off # of points plotted
def yticks_mm(ax):
    '''
    Formats the y axis ticks to millions for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    ylabels = ['{:.1f}'.format(y) + 'MM' for y in ax.get_yticks()/1000_000];
    #Force yticks (handles user interaction)
    ax.set_yticks(ax.get_yticks());
    ax.set_yticklabels(ylabels);
    return None

def yticks_k(ax):
    '''
    Formats the y axis ticks to thousands for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    ylabels = ['{:,.1f}'.format(y) + 'K' for y in ax.get_yticks()/1000];
    #Force yticks (handles user interaction)
    ax.set_yticks(ax.get_yticks());
    ax.set_yticklabels(ylabels);
    return None

def xticks_k(ax):
    '''
    Formats the x axis ticks to thousands for axes subplots.
    Returns: None
    Inputs: 
        (R) ax: AxesSubplot
    '''
    #Get yticks and format them
    xlabels = ['{:,.1f}'.format(x) + 'K' for x in ax.get_xticks()/1000];
    #Force yticks (handles user interaction)
    ax.set_xticks(ax.get_xticks());
    ax.set_xticklabels(xlabels);
    return None

###### Pretty Print Stats  ######
def stats_result(p,null_h,**kwargs):
    """
    Compares p value to alpha and outputs whether or not the null hypothesis
    is rejected or if it failed to be rejected.
    DOES NOT HANDLE 1-TAILED T TESTS
    
    Required inputs:  p, null_h (str)
    Optional inputs: alpha (default = .05), chi2, r, t, corr
    
    """
    #Get alpha value - Default to .05 if not provided
    alpha=kwargs.get('alpha',.05)
    #get any additional statistical values passed (for printing)
    t=kwargs.get('t',None)
    r=kwargs.get('r',None)
    chi2=kwargs.get('chi2',None)
    corr=kwargs.get('corr',None)
    
    #Print null hypothesis
    print(f'\n\033[1mH\u2080:\033[0m {null_h}')
    #Test p value and print result
    if p < alpha: print(f"\033[1mWe reject the null hypothesis\033[0m, p = {p} | α = {alpha}")
    else: print(f"We failed to reject the null hypothesis, p = {p} | α = {alpha}")
    #Print any additional values for reference
    if 't' in kwargs: print(f'  t: {t}')
    if 'r' in kwargs: print(f'  r: {r}')
    if 'chi2' in kwargs: print(f'  chi2: {chi2}')
    if 'corr' in kwargs: print(f'  corr: {corr}')

    return None



def get_model_stats(act,mod,pos,**kwargs):
    """
    Gets model statistics.  Only handles binary target variables.

    Parameters:
      (R) -        act: pandas series of actual target values
      (R) -        mod: pandas series of modeled target values (must be same length as act). 
      (R) -        pos: positive outcome for target variable.
      (O) -     ret_df: If True, it returns a single row dataframe with model statistics. Index is model name.
      (O) -  to_screen: If True, model statistics are printed to the screen.  Default True
    
    NOTE:  
    recall = sensitivity = true positive rate
    miss rate = false negative rate
    specificity = true negative rate    
    """
    #Get any keyword arguements and set defaults.
    # Default is to print to screen and not return None.
    ret_df = kwargs.get('ret_df',False)
    to_screen = kwargs.get('to_screen',True)
    
    #Create label list - binary confusion matrix needs positive value last
    #Get list of possible outcomes
    oth=list(act.unique())
    #remove positive value
    oth.remove(pos)
    #append postive value to end of list
    labels = oth +[pos]
    
    #run confusion matrix
    cm = confusion_matrix(act,mod,labels=labels)
    
    #If two target variables ravel cm, else break softly 
    if len(labels) == 2: 
        tn, fp, fn, tp = cm.ravel()
    else: 
        print('function cannot handle greater than 2 target variable outcomes')
        return None
    
    #Calculate all the model scores/statistics
    recall = recall_score(act,mod,pos_label=pos,zero_division=0)
    precision = precision_score(act,mod,pos_label=pos,zero_division=0)
    f1 = f1_score(act,mod,pos_label=pos,zero_division=0)
    acc = accuracy_score(act,mod)
    fnr = fn/(tp+fn)
    fpr = fp/(tn+fp)
    support_pos = tp + fn
    support_neg = fp + tn
    
    #print to screen unless kwarg set otherwise
    if to_screen:
        print(f'\033[1mModel: {mod.name}  Positive: {pos}\033[0m')
        print('\033[4mConfusion Matrix\033[0m')
        print(f'  TP: {tp}   FP: {fp}')
        print(f'  FN: {fn}   TN: {tn}')
        print('\033[4mAdditional Information\033[0m')
        print(f'      Accuracy: {acc:.3f}')
        print(f'     Precision: {precision:.3f}')
        print(f'        Recall: {recall:.3f}')
        print(f'      F1 score: {f1:.3f}')
        print(f'False neg rate: {fnr:.3f}')
        print(f'False pos rate: {fpr:.3f}')   
        print(f' Support (pos): {support_pos}')
        print(f' Support (neg): {support_neg}\n')

    ##Store results in Pandas Dataframe
    # Don't pass in df, but create a new one and concat it outside the function
    if ret_df:
        #Put stats in dictionary
        stats = {
            "Accuracy": acc,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "FNR": fnr,
            "FPR": fpr,
            "support_pos":support_pos,
            "support_neg":support_neg,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
        #convert and reshape data into dataframe, with index as model name and columns as the model statistics
        df = pd.DataFrame(data=stats,index=[mod.name])
        return df
    else: return None