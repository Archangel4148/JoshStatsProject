import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Read in data
data_frame = pd.read_csv('sleep.csv')
data_frame = data_frame.copy()

# Significant variables
# --- (All the variables: ["log_BodyWt", "log_BrainWt", "Life", "log_GP", "P", "SE", "D"])
significant_variables = ["log_BodyWt", "log_GP", "D"]

# Log the widespread variables to keep everything normal-ish
data_frame["log_BodyWt"] = np.log(data_frame["BodyWt"])
data_frame["log_BrainWt"] = np.log(data_frame["BrainWt"])
data_frame["log_GP"] = np.log(data_frame["GP"])

# Remove rows that are missing important data
clean_frame = data_frame.dropna(subset=significant_variables + ["TS"])

# Get variables for regression
x = clean_frame[significant_variables]
y = clean_frame["TS"]

# IDK, the docs told me to do this
x = sm.add_constant(x)

# Run the regression
est = sm.OLS(y, x).fit()
print(est.summary())

# Get the resulting predictions
predictions = est.predict(x)

# Sort by actual values
sort_order = np.argsort(clean_frame["TS"])
species = clean_frame["Species"]
sorted_species = [s.replace("_", " ").title() for s in species.iloc[sort_order]]
sorted_actual = clean_frame["TS"].iloc[sort_order]
sorted_predictions = predictions.iloc[sort_order]

print("Number of species: ", len(sorted_species))

# Plot
plt.figure(figsize=(12, 8))

# Plot actual
plt.plot(sorted_species, sorted_actual, "o", label="Actual Sleep (hours)")

# Plot predictions
plt.plot(sorted_species, sorted_predictions, "o", label="Predicted Sleep (hours)")

# Format plot
plt.xlabel("Species")
plt.ylabel("Sleep (hours)")
plt.xticks(rotation=90, fontsize=8, ha='center')  # ha = horizontal alignment

plt.legend()
plt.tight_layout()

# Show and save
plt.savefig("results_plot.png")
# plt.show()
