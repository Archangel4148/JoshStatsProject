import numpy as np
import pandas as pd
import statsmodels.api as sm

# Read in data
data_frame = pd.read_csv('sleep.csv')

# Remove rows with missing total sleep
clean_frame = data_frame.dropna()
clean_frame = clean_frame.copy()

# Log transform the widespread variables
clean_frame["log_BodyWt"] = np.log(clean_frame["BodyWt"])
clean_frame["log_BrainWt"] = np.log(clean_frame["BrainWt"])
clean_frame["log_GP"] = np.log(clean_frame["GP"])

# Get variables for regression
x = clean_frame[["log_BodyWt", "log_BrainWt", "Life", "log_GP", "P", "SE", "D"]]
y = clean_frame["TS"]

x = sm.add_constant(x)

# Run the regression
est = sm.OLS(y, x).fit()
print(est.summary())
