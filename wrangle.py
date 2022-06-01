import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split

#### DATA ACQUISITION ####

#Function to create database url.  Requires local env.py with host, username and password. 
# No function help text provided as we don't want the user to access it and display their password on the screen
def get_db_url(db_name,user=username,password=password,host=host):
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

#Function to get new data from Codeup server
def getNewZillowData():
    """
    Retrieves zillow dataset from Codeup DB and stores a local csv file
    Returns: Pandas dataframe
    """
    db_name= 'zillow'
    filename='zillow.csv'
    sql = """
    SELECT bedroomcnt as bed,
        bathroomcnt as bath, 
        calculatedfinishedsquarefeet as sf, 
        taxvaluedollarcnt as value, 
        yearbuilt, 
        taxamount, 
        fips
    FROM properties_2017
        JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusedesc = 'Single Family Residential';
    """
    #Read SQL from file
    df = pd.read_sql(sql,get_db_url(db_name))
    #write to disk - writes index as col 0:
    df.to_csv(filename)
    return df

#Function to get data from local file or Codeup server 
def getZillowData():
    """
    Retrieves Zillow dataset from working directory or Codeup DB. Stores a local copy if one did not exist.
    Returns: Pandas dataframe of zillow data
    """
    #Set filename
    filename = 'zillow.csv'

    if os.path.isfile(filename): #check if file exists in WD
        #grab data, set first column as index
        return pd.read_csv(filename,index_col=[0])
    else: #Get data from SQL db
        df = getNewZillowData()
    return df
##########################
##########################

#### DATA PREPARATION ####

#### DATA SPLITTING ####
def splitData(df,**kwargs):
    """
    Splits data into three dataframes
    Returns: 3 dataframes in order of train, test, validate
    Inputs:
      (R)             df: Pandas dataframe to be split
      (O -kw)  val_ratio: Proportion of the whole dataset wanted for the validation subset (b/w 0 and 1). Default .2 (20%)
      (O -kw) test_ratio: Proportion of the whole dataset wanted for the test subset (b/w 0 and 1). Default .1 (10%)
    """
    #Pull keyword arguments and set test and validation percentages of WHOLE dataset 
    val_per = kwargs.get('val_ratio',.2)
    test_per = kwargs.get('test_ratio',.1)

    #Calculate percentage we need of test/train subset
    tt_per = test_per/(1-val_per)

    #Split validate dataset off
    #returns train then test, so test_size is the second set it returns
    tt, validate = train_test_split(df, test_size=val_per,random_state=88)
    #now split tt in train and test 
    train, test = train_test_split(tt, test_size=tt_per, random_state=88)
    
    return train, test, validate

#### ZILLOW PREP ####
def prep_zillow(df,**kwargs):
  """
  Cleans and prepares the telco data for analysis.  Assumes default SQL query - with resulting columsn - was used.
  Returns: 3 dataframes in order of train, test, validate
  Inputs:
    (R) df: Pandas dataframe to be cleaned and split for analysis
    (O -kw) val_ratio: Proportion of the whole dataset wanted for the validation subset (b/w 0 and 1). Default .2 (20%)
    (O -kw) test_ratio: Proportion of the whole dataset wanted for the test subset (b/w 0 and 1). Default .1 (10%)
  """
  #Drop nulls:
  df.dropna(inplace=True)
  
  #Trim dataset
  #drop top .1% of sf
  df = df[df.sf<df.sf.quantile(.999)]
  #drop anything less than 120 sf
  df = df[df.sf>=120]
  #drop 10+ beds, 10+ baths and 10+ million
  df = df[(df.value < 10_000_000) & (df.bath < 10) & (df.bed <10)]

  #MAP fips to a county column
  df['county'] = df.fips.map({6037: 'LosAngeles_CA',6059:'Orange_CA',6111:'Ventura_CA'})
  
  #ENCODE into dummy df
  d_df = pd.get_dummies(df['county'],drop_first=True)
  #concat dummy df to the rest
  df = pd.concat([df,d_df],axis=1)
  
  #CONVERT some floats to int
  df.bed = df.bed.astype(int)
  df.yearbuilt = df.yearbuilt.astype(int)

  #DROP original fips column
  df.drop(columns='fips',inplace=True)

  #REORDER columns
  df = df.reindex(columns=['value', 'county', 'bed', 'bath', 'sf', 'yearbuilt', 'taxamount', 'Orange_CA', 'Ventura_CA'])

  #Now split the data:
  train, test, validate = splitData(df,**kwargs)

  return train, test, validate