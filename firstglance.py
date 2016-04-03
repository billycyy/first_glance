#author: Yiyun Chen
import pandas as pd
import numpy as np
from IPython.display import HTML,display
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.grid_search import RandomizedSearchCV
import time
from sklearn.feature_extraction import DictVectorizer
from operator import itemgetter

pd.set_option('display.max_rows', 15)
num_types = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'float128']

class MissingImputer(BaseEstimator, TransformerMixin):
	"""
	An inheritance class of BaseEstimator, TransformerMixin in sklearn. It can be used in sklearn.pipeline.
	It can impute the missing values in pandas dataframe.
	
	Parameters:
	
	method: string. can be "mean", "max", "median", "mode". Default is "mean".

	Return:
	A pandas dataframe. The original one will not be affected.	
	"""
	def __init__(self, method="mean"):
		self.method = method
		
	def fit(self, X, y=None):
		if self.method == "mean":
			self.values = X[get_num_col(X)].mean()
		elif self.method == "max":
			self.values = X[get_num_col(X)].max()+1
		elif self.method == "median":
			self.values = X[get_num_col(X)].median()
		elif self.method == "mode":
			self.values = X[get_num_col(X)].mode().iloc[0]
		return self
		
	def transform(self, X, y=None):
		XX = X.copy()
		XX.fillna(self.values, inplace=True)
		return XX

class CategoricalConverter(BaseEstimator, TransformerMixin):
	"""
	An inheritance class of BaseEstimator, TransformerMixin in sklearn. It can be used in sklearn.pipeline.
	It can convert categorical columns to numeric ones.
	
	Parameters:
	
	method: string. can be "dummy", "groupmean", "valuecount". Default is "dummy".
	cate_col: None or a list of string of column names.
	
	Return:
	A pandas dataframe with the categorical columns dropped. The original one will not be affected.
	"""
	def __init__(self, method="dummy", cate_cols=None):
		self.method = method
		self.cate_cols = cate_cols
		return
		
	def fit(self, X, y):
		if self.cate_cols is None:
			self.cate_cols = get_cate_col(X)
		self.values = {}
		if self.method == "dummy":
			self.dvec = DictVectorizer(sparse=False)
			self.dvec.fit((X[self.cate_cols]).to_dict('record'))
		elif self.method == "groupmean":
			for col in self.cate_cols:
				tempdict = {}
				tempvals = [val for val in X[col].unique() if str(val) != "nan"]
				for val in tempvals:
					tempdict[val] = y[(X[col] == val)].mean()
				self.values[col] = tempdict
		elif self.method == "valuecount":
			for col in self.cate_cols:
				self.values[col] = X[col].value_counts()
		return self
		
	def transform(self, X, y=None):
		XX = X.copy()
		if self.method == "dummy":
			temp_dummy = pd.DataFrame( data = self.dvec.transform((XX[self.cate_cols]).to_dict('record')), columns = self.dvec.get_feature_names(), index=XX.index)
			XX = pd.concat([temp_dummy,XX],axis=1)
		elif self.method in ["groupmean", "valuecount"]:
			for col in self.cate_cols:
				XX.loc[:,col+"_gpmean"] = XX[col].map(self.values[col])
		
		XX.drop(self.cate_cols, axis=1, inplace=True)
		return XX
		
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
	top_n: None or int. The number of highest missing value rate. Default is None, which means retaining all the records.	
	
	Return:
	
	A pandas Series of missing value rates, having column names as index, sorted in descending order.
	"""
	msratedf = df.apply(axis=0, func=lambda x: x.isnull().mean())
	msratedf = msratedf[msratedf>0]
	msratedf.sort_values(ascending=False,inplace=True)
	if top_n is not None and top_n < len(msratedf):
		msratedf = msratedf[:top_n]
	return msratedf

def get_num_of_unique(df, top_n=None):
	"""
	This function will provide number of categories for the categorical columns.
	
	Parameters:
	
	df: Pandas dataframe with all categorical columns	
	top_n: None or int. The number of highest missing value rate. Default is None, which means retaining all the records.	
	
	Return:
	
	A pandas Series of number of categories, having column names as index, sorted in descending order.
	"""
	numdf = df.apply(axis=0, func=lambda x: len(x[x.notnull()].unique()))
	numdf.sort_values(ascending=False, inplace=True)
	if top_n is not None and top_n < len(numdf):
		numdf = numdf[:top_n]
	return numdf
	
def report(grid_scores, n_top=10):
	"""
	This function will print scores from grid search results.
	
	Parameters:
	
	grid_scores: grid score from grid search of scikit-learn
	n_top: int, number of top models whose score will be shown. default is 10
	
	"""
	top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
	display(HTML("<hr>"))
	for i, score in enumerate(top_scores):
		display(HTML("<h3>"+("Model with rank: {0}".format(i + 1))+"</h3>"))
		display(HTML("<h3>"+("Mean validation score: {0:.3f} (std: {1:.3f})".format(
			  score.mean_validation_score,
			  np.std(score.cv_validation_scores)))+"</h3>"))

		display(HTML("<h3>"+("Parameters: {0}".format(score.parameters))+"</h3>"))
		display(HTML("<h3>"+" </h3>"))
	display(HTML("<hr>"))
	return 
	
def analyze_it(x, y, problem_type="infer", seed=42):
	"""
	This function will produce a report of quick analysis of a dataset.
	
	Parameters:
	x: Pandas dataframe containing all the predictors.
	y: Pandas Series containing the response.
	problem_type: string, can be "classification" "regression" "infer". default is "infer"
	seed: int,the seed to control random forest. default is 42.
	
	"""
	start_time = time.time()
	xx = x.copy()
	yy = y.copy()
	has_cate = False
	has_miss = False
	
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
	pipelist = []

	
	if len(get_num_col(xx)) > 0:
		display(HTML("<hr>"))
		display(HTML("<h2> Description of numeric columns</h2>"))		
		display_all_cols(xx[get_num_col(xx)].describe())
	
	
	if len(get_cate_col(xx)) > 0:
		has_cate = True
		display(HTML("<hr>"))
		display(HTML("<h2> Description of categorical columns</h2>"))
		ncate = get_num_of_unique(xx[get_cate_col(xx)], top_n=10)
		ncate.sort_values(inplace=True)
		ncate.plot(kind="barh", title=("Top %d number of categories" % len(ncate)))
		plt.show()
		pipelist.append(('convert',CategoricalConverter()))
		pipelist.append(('impute',MissingImputer()))
	
	

	msrate = get_missing_rate(xx, top_n=10)
	if len(msrate) > 0:
		has_miss = True
		display(HTML("<hr>"))
		display(HTML("<h2> Missing values </h2>"))	
		msrate.sort_values(inplace=True)
		msrate.plot(kind="barh", title=("Top %d missing value rates" % len(msrate)))
		plt.show()
		if not has_cate:
			pipelist.append(('impute',MissingImputer()))	
	
	display(HTML("<hr>"))
	
	if problem_type == "classification":
		display(HTML("<h2> Distribution of response</h2>"))
		yrate = yy.value_counts()/len(yy)
		yrate.plot(kind="barh")
		plt.show()
		pipelist.append(('RF',RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=seed)))
	else:
		display(HTML("<h2> Histogram of response</h2>"))
		yy.hist()
		plt.show()
		pipelist.append(('RF',RandomForestRegressor(n_estimators=50,n_jobs=-1,random_state=seed)))
	
	display(HTML("<hr>"))
	display(HTML("<h2> Benchmark using Random Forest with 50 trees</h2>"))		
	display(HTML("<h3> This is a %s problem.</h3>" % problem_type))	
	
	
	
	pipe = Pipeline(pipelist)
	paras = {}
	if has_cate:
		paras['convert__method'] = ["dummy", "groupmean", "valuecount"]
		paras['impute__method'] = ["mean","median","max","mode"]
	elif has_miss:
		paras['impute__method'] = ["mean","median","max","mode"]
	
	paras['RF__max_features'] = [0.1,0.5,0.9,'log2','sqrt']
	paras['RF__max_depth'] = [None,5,10,20]
	if problem_type == "classification":
		paras['RF__criterion'] = ['gini','entropy']
	
	if problem_type == "classification":
		cv = RandomizedSearchCV(pipe, paras, n_iter=20, scoring="roc_auc", random_state=seed)
		cv.fit(x,y)
		display(HTML("<h3> 3-fold Cross validated ROC_AUC score: %5.3f</h3>" % cv.best_score_))
		if has_cate:
			display(HTML("<h3> Best parameter for converting categorical features: %s</h3>" % cv.best_params_['convert__method']))
			display(HTML("<h3> Best parameter for missing value imputation: %s</h3>" % cv.best_params_['impute__method']))
		elif has_miss:
			display(HTML("<h3> Best parameter for missing value imputation: %s</h3>" % cv.best_params_['impute__method']))
		
	else:
		cv = RandomizedSearchCV(pipe, paras, n_iter=20, scoring="mean_squared_error", random_state=seed)
		cv.fit(x,y)
		display(HTML("<h3> 3-fold Cross validated MSE: %20.3f</h3>" % abs(cv.best_score_)))
		if has_cate:
			display(HTML("<h3> Best parameter for converting categorical features: %s</h3>" % cv.best_params_['convert__method']))
			display(HTML("<h3> Best parameter for missing value imputation: %s</h3>" % cv.best_params_['impute__method']))
		elif has_miss:
			display(HTML("<h3> Best parameter for missing value imputation: %s</h3>" % cv.best_params_['impute__method']))
	
	report(cv.grid_scores_, n_top=3)
	
	display(HTML("<h3> --- %10.2f seconds --- </h3>" % (time.time() - start_time)))
	
	
	return None
