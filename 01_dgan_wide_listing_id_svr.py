
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

# sample 5 id's from each id
sampled_ids = data_raw.groupby('cluster')['id'].apply(lambda x: x.sample(n=3, random_state=42)).reset_index(drop=True)
data_long = data_raw[data_raw['id'].isin(sampled_ids)]

# prep & view
data_long['date'] = pd.to_datetime(data_raw['date'])
data_long.glimpse()

data_long_prepared = data_long.sort_values(by=['id', 'date'])

# only select 3 (for now...)
# data_long_prepared = data_long[(data_long['id'] == "listing_id_1102") | (data_long['id'] == "listing_id_1144") | (data_long['id'] == "listing_id_3975")]


data_long_prepared \
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
min_date = pd.to_datetime(data_long_prepared['date'].min())
max_date = pd.to_datetime(data_long_prepared['date'].max())

# always add 1 to include the last day
time_line = ((max_date - min_date).days) + 1

target_length = 10
sample_length = closest_divisor(time_line, target_length)




# 1.1 PIVOT WIDER
data_wide = data_long_prepared \
    .pivot(index=["id", 'cluster'], columns="date", values="value")


# reset index and drop date column
data_wide.reset_index(inplace=True)
data_wide.columns.name = None




# 2.0 MODELING & TRAINING ----

# Train the model
model_by_listing = DGAN(DGANConfig(
   max_sequence_len = time_line,
   sample_len = sample_length,
   batch_size = 3000,
   epochs = 100,
))




# WIDE FORMAT
model_by_listing.train_dataframe(
   df = data_wide,
   example_id_column = 'id',
   attribute_columns = ['cluster'],
   discrete_columns = ['cluster']
)



# 2.1 GENERATE SYNTHETIC DATA ----
synthetic_raw = model_svr.generate_dataframe(10)
synthetic_raw


# Unique id's
# Function to add suffixes to duplicate IDs
def add_suffix_to_duplicates(df, column):
    duplicates = df[column].duplicated(keep=False)
    df['suffix'] = df.groupby(column).cumcount().astype(str)
    # df.loc[duplicates, 'suffix'] = df.loc[duplicates, column] + '_' + df.loc[duplicates, 'suffix']
    # df.drop(columns='suffix', inplace=True)
    return df

# Apply the function
synthetic_df = add_suffix_to_duplicates(df = synthetic_raw, column = 'id')



# Reshape
synthetic_long = synthetic_df \
   .melt(
      id_vars=['id', 'cluster', 'suffix'],
      var_name="date",
      value_name="value"
   ) \
    .sort_values(by = ['id', 'date'])

synthetic_long['date'] = pd.to_datetime(synthetic_long['date'])
synthetic_long['value'] = synthetic_long['value'].astype(float)

synthetic_prepared = synthetic_long

synthetic_prepared.glimpse()





synthetic_prepared \
    .groupby('id') \
    .plot_timeseries(
        date_column = 'date',  
        value_column = 'value',
        color_column = 'suffix',
        title = 'Real Data Only',
        engine = 'plotly',
        smooth = False,
        smooth_alpha = 0
    )





# VISUALIZE ----

# Function to randomly sample listings and concatenate
def sample_and_concatenate(data_raw, synthetic_prepared, num_samples=5):
    
    
    # SYNTHETIC DATA ----
    # Randomly sample unique clusters from synthetic_prepared
    sampled_ids = random.sample(synthetic_prepared['id'].unique().tolist(), num_samples)
    
    # Filter synthetic_prepared for the sampled ids
    filtered_synthetic = synthetic_prepared[synthetic_prepared['id'].isin(sampled_ids)]
    filtered_synthetic['value'] = filtered_synthetic['value'].astype(float)
    filtered_synthetic['type'] = "synthetic"
    filtered_synthetic['date'] = pd.to_datetime(filtered_synthetic['date'])
    
       
    # REAL DATA ----
    filtered_df_long = data_raw[data_raw['id'].isin(sampled_ids)]
    filtered_df_long['type'] = "real"
    filtered_df_long['suffix'] = "real"
    filtered_df_long['date'] = pd.to_datetime(filtered_df_long['date'])
    
    # Concatenate df_long and filtered_synthetic
    data_bound = pd.concat([filtered_df_long, filtered_synthetic], ignore_index=True)
    
    return data_bound


# Apply the function and get the concatenated dataframe
results_bound = sample_and_concatenate(data_raw, synthetic_prepared, num_samples=5)
results_bound.glimpse()

results_bound \
    .groupby('id') \
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



















