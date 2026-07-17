# --------------------------------------------------

# 1. Import libraries
# --------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv("data.csv")

# --------------------------------------------------
# 3. Inspect the dataset
# --------------------------------------------------

# Display the first five observations
print(df.head())

# Dataset dimensions (rows, columns)
print(df.shape)

# Variable types and missing values
df.info()

# Descriptive statistics
print(df.describe())

# --------------------------------------------------
# 4. Explore the data
# --------------------------------------------------

# Histograms
variables = ["stress", "sleep_hours", "sleep_quality", "wellbeing"]
for var in variables:
    plt.hist(df[var])
    plt.title(var)
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.savefig(f"{var}_histogram.png")
    plt.show()

# Boxplots
variables = ["stress", "sleep_hours", "sleep_quality", "wellbeing"]

for var in variables:
    print(var)
    df[var]
    plt.boxplot(df[var])
    plt.title(var)
    plt.savefig(f"{var}_boxplot.png")
    plt.show()




# --------------------------------------------------
# 5. Correlation analysis
# --------------------------------------------------
print(df[variables].corr())

# --------------------------------------------------
# 6. Data visualization
# --------------------------------------------------
predictors = ["stress", "sleep_hours", "sleep_quality"]

for pre in predictors:
    print(pre)
    sns.regplot(
    data=df,
    x=pre,
    y="wellbeing"
)
    
    plt.title(f"{pre} vs Wellbeing")
    plt.savefig(f"{pre}_wellbeing.png")

    plt.show()

# --------------------------------------------------
# 7. Multiple linear regression
# --------------------------------------------------

# Base model
model = smf.ols(
    "wellbeing ~ stress + sleep_hours + sleep_quality",
    data=df
).fit()
print(model.summary())

# Moderation/interaction by gender
model_mod_gender = smf.ols(
    "wellbeing ~ stress * gender + sleep_hours + sleep_quality",
    data=df
).fit()

print(model_mod_gender.summary())


# Moderation/interaction by age
model_mod_age = smf.ols(
    "wellbeing ~ stress * age + sleep_hours + sleep_quality",
    data=df
).fit()

print(model_mod_age.summary())

# --------------------------------------------------
# 8. Model diagnostics
# Residuals vs. Fitted Values
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")

plt.savefig("residuals_vs_fitted.png")
plt.show()


# Q-Q Plot

stats.probplot(model.resid, dist="norm", plot=plt)

plt.title("Normal Q-Q Plot")
plt.savefig("qq_plot.png")
plt.show()
# --------------------------------------------------

# --------------------------------------------------
# 9. Standardized model
scaler = StandardScaler()

df[["stress_z",
    "sleep_hours_z",
    "sleep_quality_z",
    "wellbeing_z"]] = scaler.fit_transform(
        df[["stress",
            "sleep_hours",
            "sleep_quality",
            "wellbeing"]]
)

model_z = smf.ols(
    "wellbeing_z ~ stress_z + sleep_hours_z + sleep_quality_z",
    data=df
).fit()

print(model_z.summary())

# --------------------------------------------------