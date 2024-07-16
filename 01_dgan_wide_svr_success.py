
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType




# 0.0 RAW DATA ----

# WIDE FORMAT----
data_raw = pd.read_csv("00_data/track_combined_calendar_prepared_svr.csv")

data_raw['date'] = pd.to_datetime(data_raw['date'])

data_raw.glimpse()

data_raw \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'cluster',
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
min_date = pd.to_datetime(data_raw['date'].min())
max_date = pd.to_datetime(data_raw['date'].max())

# always add 1 to include the last day
time_line = ((max_date - min_date).days) + 1

target_length = 10
sample_length = closest_divisor(time_line, target_length)




# 1.1 PIVOT WIDER
data_wide = data_raw \
    .pivot(index="cluster", columns="date", values="value")


# reset index and drop date column
data_wide.reset_index(inplace=True)
data_wide.columns.name = None





# 2.0 MODELING & TRAINING ----

# Train the model
model_svr = DGAN(DGANConfig(
   max_sequence_len = time_line,
   sample_len = sample_length,
   batch_size = 3000,
   epochs = 10000,
))


# WIDE FORMAT
model_svr.train_dataframe(
   df = data_wide,
   attribute_columns = ['cluster'],
   example_id_column = ['cluster']
)




# 2.1 GENERATE SYNTHETIC DATA ----
synthetic_df = model_svr.generate_dataframe(50)
synthetic_df


# HANDLE DUPLICATES ----
def add_suffix_to_duplicates(df, column):
    duplicates = df[column].duplicated(keep=False)
    df['suffix'] = df.groupby(column).cumcount().astype(str)
    # df.loc[duplicates, 'suffix'] = df.loc[duplicates, column] + '_' + df.loc[duplicates, 'suffix']
    # df.drop(columns='suffix', inplace=True)
    return df


# Apply the function
synthetic_df = add_suffix_to_duplicates(df = synthetic_df, column = 'cluster')


# Reshape
synthetic_long = synthetic_df \
   .melt(
      id_vars=['cluster', 'suffix'],
      var_name="date",
      value_name="value"
   )


synthetic_long['date'] = pd.to_datetime(synthetic_long['date'])
synthetic_long












# VISUALIZE ----

# Function to randomly sample listings and concatenate
def sample_and_concatenate(data_raw, synthetic_prepared, num_samples=5):
    
    # Randomly sample unique clusters from synthetic_prepared
    sampled_clusters = random.sample(synthetic_prepared['cluster'].unique().tolist(), num_samples)
    
    
    # SYNTHETIC DATA ----
    # Filter synthetic_prepared for the sampled clusters
    filtered_synthetic = synthetic_prepared[synthetic_prepared['cluster'].isin(sampled_clusters)]
    filtered_synthetic['value'] = filtered_synthetic['value'].astype(float)
    filtered_synthetic['type'] = "synthetic"
    filtered_synthetic['date'] = pd.to_datetime(filtered_synthetic['date'])
    
    
    # REAL DATA ----
    filtered_df_long = data_raw[data_raw['cluster'].isin(sampled_clusters)]
    filtered_df_long['type'] = "real"
    filtered_df_long['suffix'] = "real"
    
    
    # Concatenate df_long and filtered_synthetic
    data_bound = pd.concat([filtered_df_long, filtered_synthetic], ignore_index=True)
    
    return data_bound



# Apply the function and get the concatenated dataframe
results_bound = sample_and_concatenate(data_raw, synthetic_long, num_samples=10)


results_bound.glimpse()



results_bound \
    .groupby('cluster') \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'suffix',
        facet_ncol = 2,
        title = 'Real & Synthetic Data Comparison',
        engine = 'plotly',
        smooth = False,
        smooth_alpha = 0
    )





# MAE CALCULATION ----

# filter suffix either "real" or "0"
filtered_df = results_bound[results_bound['suffix'].isin(['real', '0'])]

# normalize
normalized_df = filtered_df.copy()
normalized_df['value'] = normalized_df.groupby('cluster')['value'].transform(lambda x: (x - x.mean()) / x.std())



# split the normalized
real_values = normalized_df[normalized_df['suffix'] == 'real']
zero_values = normalized_df[normalized_df['suffix'] == '0']


# blank results
results = []


# MAE for each cluster
clusters = normalized_df['cluster'].unique()
for cluster in clusters:
    cluster_real_values = real_values[real_values['cluster'] == cluster]['value']
    cluster_zero_values = zero_values[zero_values['cluster'] == cluster]['value']
    
    # confirm matching pairs of real and 0 values
    if len(cluster_real_values) == len(cluster_zero_values):
        mae = mean_absolute_error(cluster_real_values, cluster_zero_values)
        results.append({'Cluster': cluster, 'MAE': mae})



results_df = pd.DataFrame(results)

results_df






