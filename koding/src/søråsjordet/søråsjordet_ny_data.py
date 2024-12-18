import os
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime as dt
import time
from itertools import groupby

# Paths
programmering_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\'
mcclear_path = programmering_path + 'måledata/temp/mcclear_søråsjordet_juni2019.csv'
outpath = programmering_path + 'måledata/analyser/søråsjordet'

raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=1)
raw_cs_df.set_index('TIMESTAMP', inplace=True)
# old_df = raw_cs_df.loc[start_date:end_date]
# cs_df = old_df.drop('Observation period', axis=1)

# Henter innstrålingsdata, her fra Søråsjordet
dir_file_path = programmering_path + '/måledata/søråsjordet/NMBU_BIOKLIM_10min_Juni-2020.xlsx'
søråsjordet_juni = pd.read_excel(dir_file_path)

søråsjordet_juni['date'] = søråsjordet_juni['DATO-TID'].dt.date
søråsjordet_juni['time'] = søråsjordet_juni['DATO-TID'].dt.time.astype(str)


def heat_map_improved_one(dataframe1):

    # y_ticks = ['00:00:00', '03:00:00', '06:00:00', '09:00:00', '12:00:00', '15:00:00', '18:00:00', '21:00:00']

    fig, ax1 = plt.subplots(1,1)
    ax1 = sns.heatmap(dataframe1, vmin=0, vmax=1200, cmap='plasma', ax=ax1)
    # ax1.set_yticks(y_ticks)
    plt.show()

heat_map_df = søråsjordet_juni.pivot(index='time', columns='date', values='GLOB')
heat_map_improved_one(heat_map_df)

raw_cs_df['DATO-TID'] = raw_cs_df.index
søråsjordet_juni['DATO-TID'] = pd.to_datetime(søråsjordet_juni['DATO-TID'])



full_df = søråsjordet_juni.merge(raw_cs_df, on='DATO-TID', how = 'left')
full_df.loc[full_df['GLOB'] < 0, 'GLOB'] = 0

active_df = full_df[full_df['ghi_clear']>0]
active_df['oie'] = active_df['GLOB'] > active_df['ghi_clear']

def get_max_measured_and_expected(group):
    if group['SWin'].notna().any():
        max_idx = group['SWin'].idxmax()
        max_value = group.loc[max_idx, 'SWin']
        expected_value = group.loc[max_idx, 'ghi_clear']
        return pd.Series([max_value,expected_value])
    else:
        return pd.Series([np.nan, np.nan])



fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(full_df['DATO-TID'], full_df['GLOB'], label = 'målt')
ax2.plot(full_df['DATO-TID'], full_df['ghi_clear'], label = 'modell')
plt.show()