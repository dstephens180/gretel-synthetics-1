
# LIBRARIES ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pytimetk as tk
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import string
import random

from math import gcd
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType




# 0.0 RAW DATA ----

# LONG FORMAT
data_raw = pd.read_csv("00_data/track_combined_calendar_synthetic_long_svr.csv")
cluster_id_raw = pd.read_csv("00_data/cluster_row_number_svr.csv")

# sample 5 id's from each cluster
sampled_ids = data_raw.groupby('cluster')['id'].apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
data_long = data_raw[data_raw['id'].isin(sampled_ids)]

# prep & view
data_long['date'] = pd.to_datetime(data_raw['date'])
data_long.glimpse()


data_long \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'id',
        title = 'Real Data Only',
        engine = 'plotly',
        smooth = False,
        smooth_alpha = 0
    )






# 1.0 DATA PREP ----

# closest divisor to target function
def closest_divisor(n, target):
    closest = 1
    for i in range(1, n + 1):
        if n % i == 0 and abs(i - target) < abs(closest - target):
            closest = i
    return closest



# sample length calculation
min_date = pd.to_datetime(data_long['date'].min())
max_date = pd.to_datetime(data_long['date'].max())

# always add 1 to include the last day
time_line = ((max_date - min_date).days) + 1

target_length = 10
sample_length = closest_divisor(time_line, target_length)






# 2.0 MODELING & TRAINING ----
data_long_prepared = pd.merge(data_long, cluster_id_raw, left_on = "cluster", right_on = "cluster", how = "left")
data_long_prepared = data_long_prepared.drop(columns=["cluster", "cluster_number"])





# Train the model
model_svr = DGAN(DGANConfig(
   max_sequence_len = time_line,
   sample_len = sample_length,
   batch_size = 3000,
   epochs = 100,
))





### YOU LEFT OFF HERE.  CAN'T GET PASSED THE "EXAMPLE_ID_COLUMN" AND "TIME_COLUMN" ISSUE.
# LONG FORMAT
data_long_filtered = data_long_prepared[(data_long_prepared['id'] == "listing_id_1102") | (data_long_prepared['id'] == "listing_id_3975")]
data_long_filtered.glimpse()


data_long_filtered['date'] = data_long_filtered['date'].astype('int64') // 10**9
data_long_filtered

model_svr.train_dataframe(
   df = data_long_filtered,
#    df_style = "long",
   time_column= "date",
#    attribute_columns = ['id'],
   example_id_column = ['id']
)




# 2.1 GENERATE SYNTHETIC DATA ----
synthetic_df = model_svr.generate_dataframe(3)
synthetic_df


# Reshape
synthetic_long = synthetic_df \
   .melt(
      id_vars=['cluster'],
      var_name="date",
      value_name="value"
   )

synthetic_long






# HANDLE DUPLICATES ----

# Function to generate suffixes
def generate_suffixes():
    single_letters = list(string.ascii_lowercase)
    multi_letters = [a+b for a in single_letters for b in single_letters]
    return single_letters + multi_letters

# Function to add suffixes to duplicates
def add_suffix(group):
    counts = group['cluster'].value_counts()
    suffixes = counts[counts > 1].index
    all_suffixes = generate_suffixes()
    suffix_dict = {cluster: iter(all_suffixes) for cluster in suffixes}
    
    def suffix_cluster(cluster):
        if cluster in suffix_dict:
            try:
                return f"{cluster}_{next(suffix_dict[cluster])}"
            except StopIteration:
                raise ValueError(f"Ran out of suffixes for {cluster}")
        return cluster
    
    group['cluster'] = group['cluster'].apply(suffix_cluster)
    return group


# Apply the function to each group
synthetic_long['date'] = pd.to_datetime(synthetic_long['date'])
# synthetic_prepared = synthetic_long.groupby('date').apply(add_suffix).reset_index(drop=True)

synthetic_prepared = synthetic_long







# VISUALIZE ----

# Function to randomly sample listings and concatenate
def sample_and_concatenate(data_raw, synthetic_prepared, num_samples=5):
    
    # Randomly sample unique clusters from synthetic_prepared
    sampled_clusters = random.sample(synthetic_prepared['cluster'].unique().tolist(), num_samples)
    
    # Filter synthetic_prepared for the sampled clusters
    filtered_synthetic = synthetic_prepared[synthetic_prepared['cluster'].isin(sampled_clusters)]
    filtered_synthetic['value'] = filtered_synthetic['value'].astype(float)
    filtered_synthetic['type'] = "synthetic"
    
    
    filtered_df_long = data_raw[data_raw['cluster'].isin(sampled_clusters)]
    filtered_df_long['type'] = "real"
    
    # Concatenate df_long and filtered_synthetic
    data_bound = pd.concat([filtered_df_long, filtered_synthetic], ignore_index=True)
    
    return data_bound


# Apply the function and get the concatenated dataframe
results_bound = sample_and_concatenate(data_raw, synthetic_prepared, num_samples=3)


results_bound.glimpse()



results_bound \
    .groupby('cluster') \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'type',
        title = 'Real & Synthetic Data Comparison',
        engine = 'plotly',
        smooth = False,
        smooth_alpha = 0
    )




















