import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


def annotate_points(x_vals, y_vals, indices, labels, offset=(-6, 6)):
    for i in indices:
        plt.annotate(
            labels[i].replace("_", " ").title(),
            xy=(x_vals[i], y_vals[i]),
            xytext=offset,
            textcoords='offset points',
            fontsize=8,
            ha='right' if offset[0] < 0 else 'left',
            arrowprops=dict(
                arrowstyle='-',
                color='gray',
                lw=0.5,
                alpha=0.6
            )
        )


if __name__ == '__main__':

    # Read in data
    data_frame = pd.read_csv('sleep.csv')

    # Significant variables
    # --- (All the variables: ["log_BodyWt", "log_BrainWt", "Life", "log_GP", "P", "SE", "D"])
    significant_variables = ["GP", "SE"]

    # Log the widespread variables to keep everything normal-ish
    data_frame["log_BodyWt"] = np.log(data_frame["BodyWt"])
    data_frame["log_BrainWt"] = np.log(data_frame["BrainWt"])
    data_frame["log_GP"] = np.log(data_frame["GP"])

    # Remove rows that are missing important data
    clean_frame = data_frame.dropna(subset=significant_variables + ["TS"]).reset_index(drop=True)

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

    # =========== RESULTS PLOT ===========

    # Plot
    plt.figure(figsize=(12, 8))

    # Plot actual
    plt.plot(sorted_species, sorted_actual, "o", label="Actual Sleep (hours)")

    # Plot predictions
    plt.plot(sorted_species, sorted_predictions, "o", label="Predicted Sleep (hours)")

    # Format plot
    plt.xlabel("Species")
    plt.ylabel("Sleep (hours)")
    plt.xticks(rotation=90, fontsize=8)  # Rotate the names so you can read them

    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig("results/results_plot.png")

    # =========== RESIDUAL PLOTS ===========

    # Getting residual data
    influence = est.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks_distance = influence.cooks_distance[0]
    fitted_values = est.fittedvalues
    residuals = est.resid

    # Create figure
    plt.figure(figsize=(12, 10))

    # ====== Residuals vs. Fitted ======
    plt.subplot(2, 2, 1)
    plt.axhline(0, color='gray', linestyle='dotted', linewidth=1, alpha=0.6)
    plt.scatter(fitted_values, residuals)
    smoothed = lowess(residuals, fitted_values)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", linestyle="--", alpha=0.6)

    # Annotate top 3
    top_idx = np.argsort(np.abs(residuals))[-3:]
    annotate_points(fitted_values, residuals, top_idx, species)

    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Fitted")

    # ====== Q-Q plot ======
    plt.subplot(2, 2, 2)
    qq_ax = plt.gca()
    sm.qqplot(standardized_residuals, line="s", fit=True, ax=qq_ax)

    # Customize the fit line
    ref_line = qq_ax.get_lines()[1]  # Line[0] is the scatter, line[1] is the 45-degree reference
    ref_line.set_linestyle('--')
    ref_line.set_alpha(0.6)
    ref_line.set_color('gray')  # Optional: use gray or red as in R
    plt.ylabel("Standardized Residuals")
    plt.title("Q-Q Residuals")

    # ====== Scale-Location plot ======
    plt.subplot(2, 2, 3)
    scale_y = np.sqrt(np.abs(standardized_residuals))
    plt.scatter(fitted_values, scale_y)
    smoothed_scale = lowess(scale_y, fitted_values)
    plt.plot(smoothed_scale[:, 0], smoothed_scale[:, 1], color="red", linestyle="--", alpha=0.6)

    # Annotate top 3
    top_idx = np.argsort(scale_y)[-3:]
    annotate_points(fitted_values, scale_y, top_idx, species)

    plt.xlabel("Fitted Values")
    plt.ylabel("√(|Standardized Residuals|)")
    plt.title("Scale-Location")

    # ====== Residuals vs. Leverage ======
    plt.subplot(2, 2, 4)
    plt.axhline(0, color='gray', linestyle='dotted', linewidth=1, alpha=0.6)
    plt.scatter(leverage, standardized_residuals)
    smoothed_leverage = lowess(standardized_residuals, leverage)
    plt.plot(smoothed_leverage[:, 0], smoothed_leverage[:, 1], color="red", linestyle="--", alpha=0.6)
    plt.xlabel("Leverage")
    plt.ylabel("Standardized Residuals")
    plt.title("Residuals vs. Leverage")
    ymin, ymax = plt.ylim()  # Get the current y bounds

    # Annotate top 3
    top_idx = np.argsort(cooks_distance)[-3:]
    annotate_points(leverage, standardized_residuals, top_idx, species)

    # Cook's distance lines
    n = x.shape[0]
    p = x.shape[1]
    thresholds = [0.5]

    leverage_range = np.linspace(min(leverage), max(leverage), 100)

    for d in thresholds:
        bound = np.sqrt(d * p * (1 - leverage_range) / leverage_range)
        plt.plot(leverage_range, bound, ls='--', color='gray', alpha=0.7, label=f"Cook's Distance: {d}")
        plt.plot(leverage_range, -bound, ls='--', color='gray', alpha=0.7)

    plt.ylim(ymin, ymax)  # Reset the y bounds (ignore Cook's distance lines)
    plt.legend()

    # ===================================

    # Show and save
    plt.tight_layout()
    plt.savefig("results/residual_plots.png")
    plt.show()
