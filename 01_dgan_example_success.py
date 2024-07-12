
# LIBRARIES ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pytimetk as tk
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from math import gcd

import string
import random

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType



# 0.0 RAW DATA ----
time_line = 451


# closest divisor to target function
def closest_divisor(n, target):
    closest = 1
    for i in range(1, n + 1):
        if n % i == 0 and abs(i - target) < abs(closest - target):
            closest = i
    return closest



# sample length calculation
target_length = 15
sample_length = closest_divisor(time_line, target_length)


df = pd.DataFrame(500 + np.random.random(size=(3, time_line)) * 1000)
df.columns = pd.date_range("2022-01-01", periods=time_line)





# PREPARE DF
df_prepared = df
df_prepared['listing_id'] = range(1, len(df_prepared)+1)
df_prepared['listing_id'] = df_prepared.apply(lambda row: f'listing_{row["listing_id"]}', axis = 1)



# original df long
df_long = df_prepared \
   .melt(
      id_vars=['listing_id'],
      var_name="date",
      value_name="value"
   )


df_long \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'listing_id',
        title = 'Real Data Only',
        engine = 'plotly',
        smooth = False,
        smooth_alpha = 0
    )




# 2.0 MODELING & TRAINING ----

# Train the model
model = DGAN(DGANConfig(
   max_sequence_len = time_line,
   sample_len = sample_length,
   batch_size = 3000,
   epochs = 3000,  # For real data sets, 100-1000 epochs is typical
))


model.train_dataframe(
   df = df_prepared,
   attribute_columns = ['listing_id'],
   example_id_column = ['listing_id']
)


# Generate synthetic data
synthetic_df = model.generate_dataframe(3)
synthetic_df





# RESHAPE ----
synthetic_long = synthetic_df \
   .melt(
      id_vars=['listing_id'],
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
def sample_and_concatenate(df_long, synthetic_prepared, num_samples=5):
    # Randomly sample unique listing_ids from synthetic_prepared
    sampled_listing_ids = random.sample(synthetic_prepared['listing_id'].unique().tolist(), num_samples)
    
    # Filter synthetic_prepared for the sampled listing_ids
    filtered_synthetic = synthetic_prepared[synthetic_prepared['listing_id'].isin(sampled_listing_ids)]
    filtered_synthetic['value'] = filtered_synthetic['value'].astype(float)
    filtered_synthetic['type'] = "synthetic"
    
    
    filtered_df_long = df_long[df_long['listing_id'].isin(sampled_listing_ids)]
    filtered_df_long['type'] = "real"
    
    # Concatenate df_long and filtered_synthetic
    data_bound = pd.concat([filtered_df_long, filtered_synthetic], ignore_index=True)
    
    return data_bound


# Apply the function and get the concatenated dataframe
results_bound = sample_and_concatenate(df_long, synthetic_prepared, num_samples=3)


results_bound.glimpse()



results_bound \
    .groupby('listing_id') \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'type',
        title = 'Real & Synthetic Data Comparison',
        engine = 'plotly',
        smooth = False,
        smooth_alpha = 0
    )




















