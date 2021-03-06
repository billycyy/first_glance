# First Glance
A python module used to produce a report and initial benchmark for datasets..


#How to use
Make sure firstglance.py is in the same directory of your python code.
Please run this module in IPython notebook.

```bash
$ import firstglance as fg
$ fg.analyze_it(x,y)
```

This will automatically produce a report. The contents include missing value rate, description of numeric and categorical features, distribution of response and cross validated benchmark by simple Random Forest. The methods used to deal with imputation of missing values and converting categorical features will be determined by grid search.

# Demo
Please go to [demo1](http://nbviewer.jupyter.org/github/billycyy/first_glance/blob/master/demo_first_glance.ipynb) and [demo2](http://nbviewer.jupyter.org/github/billycyy/first_glance/blob/master/demo_first_glance_2.ipynb) to see the demo in IPython notebook. (Nbviewer has nicer format.)
If these two link don't work, plz refer to [demo1](/demo_first_glance.ipynb) and [demo2](/demo_first_glance_2.ipynb) directly.
