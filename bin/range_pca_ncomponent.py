import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys

#JJE, 04182023
#show PCA and Logistic Regression over many components and plot 
# principle components and test scoring classification accuracies.
#Input is a CSV of sample columns and data values in rows.  Unique row names 
# are in column 1 and each sample in columns thereafter. Also required 
# is a second CSV of row names in column 1 and classification labels in column 2.
#Header is expected in both files, but isn't required.  Number of rows in data 
# file must be same as number of rows in labels file

seed = 42#random seed

try:
	dtacsv = sys.argv[1]
	labelcsv = sys.argv[2]
	
	try:
		figurefn = sys.argv[3]
	except:
		figurefn = None

except:
	message = f"usage: ml_pca_logisticregress.py data.csv label.csv figure_file_filename(optional, default launches browser instead of writing to file))\n"
	sys.stderr.write(message)
	exit(1)

dta = pd.read_csv(dtacsv)
labels = pd.read_csv(labelcsv)

#error if number of rows in data file != number of rows in label file
if len(dta[dta.columns[0]].values) != len(labels[labels.columns[0]].values):
	message = "ERROR: number of rows in data file not equal to number of rows in labels file.\n"
	raise Exception(message)

#PCA
# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()

# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# set the tolerance to a large value to make the example faster
logistic = LogisticRegression(max_iter=10000, tol=0.1)

#create Pipeline
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])

# Parameters of pipelines to iterate over
param_grid = {
    "pca__n_components": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "logistic__C": np.logspace(-4, 4, 4),
}

#cross validation
kf = KFold(n_splits=6, random_state=seed, shuffle=True)

#param tuning
search = GridSearchCV(pipe, param_grid, n_jobs=2, cv=kf)

#columns
search.fit(dta, labels.values.ravel())

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Plot the PCA spectrum
pca.fit(dta)

#Plot 1: PCA
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(
    np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
)
ax0.set_ylabel("PCA explained variance ratio")

ax0.axvline(
    search.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components chosen",
)
ax0.legend(prop=dict(size=12))

#Plot 2: PCA followed by Logistic Regression
# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)

print(results.sort_values(by=["param_pca__n_components", "param_logistic__C"], ascending=True)[['param_logistic__C', 'param_pca__n_components', 'mean_test_score', 'std_test_score', 'rank_test_score']])

components_col = "param_pca__n_components"
best_clfs = results.groupby(components_col)[
    [components_col, "mean_test_score", "std_test_score"]
].apply(lambda g: g.nlargest(1, "mean_test_score"))
ax1.errorbar(
    best_clfs[components_col],
    best_clfs["mean_test_score"],
    yerr=best_clfs["std_test_score"],
)
ax1.set_ylabel("Classification accuracy (val)")
ax1.set_xlabel("n_components")

plt.xlim(-1, 70)

plt.tight_layout()

if figurefn is not None:
	plt.savefig(figurefn)
else:
	plt.show()
	
exit()
