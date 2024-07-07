import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
#unsupervised k-means cluster number determination
#find and plot clustering inertia over a range in cluster number
# for an inputted table of data (sample per column, feature per row)
#input table file assumed tab-delimited unless ends in ".csv" in which it parses CSV
#NOTE: column name header inferred.  row names assumed not in 1st col.  if row names in 1st column, 
# use any value as 2nd argument to enable index from column 1 (optional)
#note maximum cluster number is set below
#note: data is not scaled or normalized, so preprocess prior to input to this tool
#jje 07072024
max_num_cluster = 20

try:
	dtafile = sys.argv[1]
	
	try:
		sys.argv[2]
		indexcol = 0#use 1st column as row name index
	except:
		indexcol = None#no row names in column, index by number
except:
	message = f"usage: kmeans_range_cluster_intertia.py data_table.csv  do_assume_1st_col_is_row_names(optional, any value, default=False))\n"
	sys.stderr.write(message)
	exit(1)

#read data
delim = "," if dtafile.endswith(".csv") else "\t"#default tab-delimited unless ends in .csv
	
df = pd.read_table(dtafile, sep=delim, index_col=indexcol)

ks = range(1, max_num_cluster)#range of number of clusters to generate clustering inertia
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k, n_init=100)
    
    # Fit model to samples
    model.fit(df)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
    
# Plot cluster number vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
