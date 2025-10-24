# %%
# import packages
import numpy as np
import pandas as pd


# %%
# Load the `cville_weather.csv` data. This includes data from Jan 4, 2024 to Feb 2, 2025. 
# Are there any missing data issues?
weather = pd.read_csv('cville_weather.csv')
weather





# %%
# Based on the precipitation variable, `PRCP`, make a new variable called `rain` 
# that takes the value 1 if `PRCP`>0 and 0 otherwise.
weather['rain'] = [1 if x > 0 else 0 for x in weather['PRCP']]
weather



# %%
# Build a two-state Markov chain over the states 0 and 1 for the `rain` variable. 




# %%
# For your chain from c, how likely is it to rain if it was rainy yesterday? 
# How likely is it to rain if it was clear yesterday?




# %% 
# Starting from a clear day, forecast the distribution. 
# How quickly does it converge to a fixed result? What if you start from a rainy day?




# %%
# Conditional on being rainy, plot a KDE of the `PRCP` variable.






# %%
# Describe one way of making your model better for forecasting and simulation the weather.