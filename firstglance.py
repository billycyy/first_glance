import pandas as pd
import numpy as np
from IPython.display import HTML,display
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 15)
num_types = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'float128']

def get_num_col(df):
	"""
	This function will return a list of column names that have numerical values.
	
	Parameters:
	
	df: Pandas dataframe
	
	Return:
	
	A list of column names that have numerical values
	"""
	types = df.dtypes
	cols = df.columns
	return [col for col in cols if str(types[col]) in num_types]

def get_cate_col(df):
	"""
	This function will return a list of column names that have categorical values.
	
	Parameters:
	
	df: Pandas dataframe
	
	Return:
	
	A list of column names that have categorial values
	"""
	types = df.dtypes
	cols = df.columns	
	return [col for col in cols if types[col] == object]	
	
def get_bool(df):
	"""
	This function will return a list of column names that have boolean values.
	
	Parameters:
	
	df: Pandas dataframe
	
	Return:
	
	A list of column names that have boolean values
	"""
	types = df.dtypes
	cols = df.columns	
	return [col for col in cols if types[col] == bool]
	
def bool_2_int(df):
	"""
	This function will convert the boolean columns to numerical ones.
	
	Parameters:
	
	df: Pandas dataframe
	
	Note: This function will modify the original dataframe! If you don't want, please pass in a copy of that dataframe.
	"""
	bool_cols = get_bool(df)
	df.loc[:,bool_cols] = df.loc[:,bool_cols].astype(int)
	return None
	
def display_all_cols(df):
	"""
	This function will display all the columns of a dataframe.
	
	Parameters:
	
	df: Pandas dataframe
	"""
	pd.set_option('display.max_columns', None)	
	display(df)
	pd.set_option('display.max_columns', 20)
	return None

def get_missing_rate(df, top_n=None):
	"""
	This function will provide missing value rates for all the columns having missing values.
	
	Parameters:
	
	df: Pandas dataframe	
	
	Return:
	
	A pandas Series of missing value rates, having column names as index, sorted in descending order.
	"""
	msratedf = df.apply(axis=0, func=lambda x: x.isnull().mean())
	msratedf = msratedf[msratedf>0]
	msratedf.sort(ascending=False,inplace=True)
	if top_n is not None and top_n < len(msratedf):
		msratedf = msratedf[:top_n]
	return msratedf
	
def analyze_it(x,y,problem_type="infer"):

	xx = x.copy()
	yy = y.copy()
	
	if str(yy.dtype) not in num_types:
		raise ValueError('The type of y is not numeric!')
	if yy.isnull().sum() > 0:
		raise ValueError('Y has missing values!')	
	
	cols = xx.columns
	types = xx.dtypes
	not_type = [col for col in cols if not (str(types[col]) in num_types or types[col] == object or types[col] == bool)]
	
	if len(not_type) > 0:
		raise ValueError('Columns: '+str(not_type)+' have types other than numeric, categorical(string) and boolean.')
	
	if problem_type == "infer":
		if len(yy.unique()) > 30:
			problem_type = "regression"
		else:
			problem_type = "classification"
	
	display(HTML("<h1 style='text-align:center'> First Glance</h1>"))
	bool_2_int(xx)
	msrate = get_missing_rate(xx,top_n=10)
	if len(msrate) > 0:
		display(HTML("<hr>"))
		display(HTML("<h2> Missing values </h2>"))	
		msrate.sort(inplace=True)
		msrate.plot(kind="barh", title=("Top "+str(len(msrate))+" missing value rates"))
		plt.show()
	
	if len(get_num_col(xx)) > 0:
		display(HTML("<hr>"))
		display(HTML("<h2> Description of numeric columns</h2>"))		
		display_all_cols(xx[get_num_col(xx)].describe())
	
	if len(get_cate_col(xx)) > 0:
		display(HTML("<hr>"))
		display(HTML("<h2> Description of categorical columns</h2>"))
		
		
		
		
		

	
	
	






	return