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

# Timer
start_time = time.time()
# Angir startpunkt og sluttpunkt for data som ønskes behandlet
start = '2023-05-07 00:00:00'
end = '2023-05-07 23:59:50'

# Finner solvinkler i tidsrommet
alt = 1222
latitude = 60.593531
longitude = 7.524547
# solar_positions = pvlib.solarposition.get_solarposition()

# Lager paths først
programmering_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\'
mcclear_path = programmering_path + 'måledata/temp/mcclear_finse_2023.csv'
outpath = programmering_path + 'måledata/analyser/finse'

# Finner clear sky model for plots
raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
cs_full_df = raw_cs_df[['TIMESTAMP', 'ghi_extra', 'ghi_clear']]
cs_df = cs_full_df[(cs_full_df['TIMESTAMP'] >= start) & (cs_full_df['TIMESTAMP'] <= end)]

# Henter innstrålingsdata, her fra Finse
finse_22_file_path = programmering_path + '/måledata/Finse/Finseflux_level1_10sec_2023.csv'
f_22_raw_df = pd.read_csv(finse_22_file_path)
f_22_df = f_22_raw_df[['TIMESTAMP', 'SWin', 'SWout', 'snow_depth']]
f_22_limited_df = f_22_df[(f_22_df['TIMESTAMP'] >= start) & (f_22_df['TIMESTAMP'] <= end)]

f_22_limited_df = f_22_limited_df.copy()
cs_df = cs_df.copy()

# Lager en liste med date-objekter over hvilke dager å se på
f_22_limited_df['TIMESTAMP'] = pd.to_datetime(f_22_limited_df['TIMESTAMP'])
f_22_limited_df['date'] = f_22_limited_df['TIMESTAMP'].dt.date
f_22_limited_df['time'] = f_22_limited_df['TIMESTAMP'].dt.time.astype(str)
f_22_limited_df.set_index('TIMESTAMP', inplace=True)
f_22_reset_df = f_22_limited_df.reset_index()

cs_df['TIMESTAMP'] = pd.to_datetime(cs_df['TIMESTAMP'])
cs_df['TIMESTAMP'] = cs_df['TIMESTAMP'].dt.tz_localize(None)
# cs_df['date'] = cs_df['TIMESTAMP'].dt.date
# cs_df['time'] = cs_df['TIMESTAMP'].dt.time.astype(str)
cs_df.set_index('TIMESTAMP', inplace=True)
cs_reset_df = cs_df.reset_index()

merged_df = pd.merge_asof(f_22_reset_df, cs_reset_df, on='TIMESTAMP')
merged_df.set_index('TIMESTAMP', inplace=True)

# merged_df['snow_depth'] = pd.to_numeric(merged_df['snow_depth'], errors='coerce')
merged_df['snow_depth'] = merged_df['snow_depth'].mask(merged_df['snow_depth'] > 150)
merged_df['snow_depth'] = merged_df['snow_depth'].mask(merged_df['snow_depth'] < -0.2)

# Calculates albedo, and sets negative value for measurements not applicable
merged_df['albedo'] = merged_df['SWout']/merged_df['SWin']
merged_df['albedo'] = merged_df['albedo'].where((merged_df['SWout'] > 0) &
                                                (merged_df['SWin'] > 0) &
                                                (merged_df['albedo'] <= 1), np.nan)

unique_days = merged_df['date'].unique()
unique_time = merged_df['time'].unique()

# Old method for creating snow_presence flag
# snow_presence = []
#
# for date in unique_days:
#     daily_data = merged_df[merged_df['date'] == date]
#     has_snow = daily_data['snow_depth'].dropna().gt(1.00).any()
#     snow_presence.append({'date': date, 'snow': has_snow})
#
# snow_presence_df = pd.DataFrame(snow_presence)

snow_depth_treshold = 1.0
percentage_treshold = 0.95

def calculate_snow_flag(group):
    valid_readings = group['snow_depth'].notna().sum()
    above_treshold = (group['snow_depth'] > snow_depth_treshold).sum()
    percentage_above_treshold = (above_treshold / valid_readings) if valid_readings > 0 else 0
    return pd.Series({'snow_flag': percentage_above_treshold > percentage_treshold})

snow_presence_df = merged_df.groupby('date', group_keys=False).apply(calculate_snow_flag, include_groups=False).reset_index()

active_df = merged_df[merged_df['ghi_clear'] > 0]
active_df = active_df.copy()
active_df['oie'] = active_df['ghi_clear'] < active_df['SWin']

def get_max_measured_and_expected(group):
    if group['SWin'].notna().any():
        max_idx = group['SWin'].idxmax()
        max_value = group.loc[max_idx, 'SWin']
        expected_value = group.loc[max_idx, 'ghi_clear']
        return pd.Series([max_value,expected_value])
    else:
        return pd.Series([np.nan, np.nan])

max_active_df = active_df.groupby('date', group_keys=False).apply(get_max_measured_and_expected, include_groups=False).reset_index()
max_active_df.columns = ['date', 'max_value', 'expected_max_value']
max_active_df['max_ratio'] = max_active_df['max_value']/max_active_df['expected_max_value']

simple_stats_active_df = active_df.groupby('date')['SWin'].agg(['median','mean','sum']).reset_index()
count_full_df = merged_df.groupby('date')['SWin'].count().reset_index()
count_full_df.columns = ['date', 'count']
stats_active_df = max_active_df.merge(simple_stats_active_df, on='date', how='left')
stats_active_df = stats_active_df.merge(count_full_df, on='date', how='left')
stats_active_df.columns = ['date', 'max_value', 'expected_max_value', 'max_ratio', 'median', 'mean', 'sum_measured', 'count']

stats_active_df['health'] = stats_active_df['count']/(24*60*6)
stats_model_df = active_df.groupby('date')['ghi_clear'].sum().reset_index()
stats_model_df.columns = ['date', 'sum_model']


def calculate_oie_lengths (group):
    series = group['oie']
    oie_lengths = [sum(1 for _ in g) for k, g in groupby(series) if k]
    return pd.Series({'oie_lengths': oie_lengths})

oie_lengths_by_day = active_df.groupby('date', group_keys=False).apply(calculate_oie_lengths, include_groups=False).reset_index()
oie_lengths_by_day.columns = ['date', 'oie_lengths']

oie_count = active_df.groupby('date')['oie'].sum().reset_index()
oie_count.columns = ['date', 'oie_counts']

daily_stats_df = snow_presence_df.merge(oie_count, on='date', how='left')
daily_stats_df = daily_stats_df.merge(oie_lengths_by_day, on='date', how='left')
daily_stats_df = daily_stats_df.merge(stats_active_df, on='date', how='left')
daily_stats_df = daily_stats_df.merge(stats_model_df, on='date', how='left')
daily_stats_df['measured/expected'] = daily_stats_df['sum_measured']/daily_stats_df['sum_model']
daily_stats_df['oie_counts'] = daily_stats_df['oie_counts'].fillna(0).astype(int)


def heat_map(irr_data, cs_data, time, frequency, num_days):
    """
    Creates two heat maps, one for measured and one for clear sky values.

    Parameters:
    ----------
    irr_data : Series
    cs_data : Series
    time : Series
    frequency : String
    num_days : int

    Returns:
    -------
    Heat map plot

    """

    seconds = 86400
    ten_seconds = seconds//10
    minutes = seconds//60
    hours = seconds//3600

    if frequency == 'ten_seconds':
        frequency = ten_seconds
        f = 10
    elif frequency == 'minutes':
        frequency = minutes
        f = 60
    elif frequency == 'seconds':
        frequency = seconds
        f = 1
    elif frequency == 'hours':
        frequency = hours
        f = 3600
    else:
        raise Exception('Frequency must be \'ten_seconds\' or \'minutes\' or \'seconds\' or \'hours\'.')


    time_of_day = np.array([pd.Timestamp(t) for t in time])
    time_numeric = [(t - time_of_day[0]).total_seconds() for t in time_of_day]

    irr_data_arr = np.empty((frequency, num_days))
    cs_data_arr = np.empty((frequency, num_days))

    n = (time_of_day[0].hour*3600 + time_of_day[0].minute*60 + time_of_day[0].second) /f
    n = int(n)
    for i in range(num_days):
        for j in range(ten_seconds):
            # Check if there still is data in irr_data and cs_data
            if n < len(irr_data) and n < len(cs_data):
                irr_data_arr[j,i] = irr_data[n]
                cs_data_arr[j,i] = cs_data[n]
                n += 1
            else:
                # If dataset ends before assigned periods, break out
                break

    diff_data_arr = irr_data_arr - cs_data_arr

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1 = sns.heatmap(irr_data_arr, vmin=0, vmax=1200, cmap='plasma', ax=ax1)
    # ax1.set_ylim(0,24)
    # ax1.set_ylabel''
    #plt.draw()
    #plt.tight_layout()

    ax2 = sns.heatmap(cs_data_arr, vmin=0, vmax=1200, cmap='plasma', ax=ax2)

    ax3 = sns.heatmap(diff_data_arr, vmin=0, vmax=200, cmap='plasma', ax=ax3)

    plt.show()


def heat_map_improved(dataframe1, dataframe2):

    # y_ticks = ['00:00:00', '03:00:00', '06:00:00', '09:00:00', '12:00:00', '15:00:00', '18:00:00', '21:00:00']

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1 = sns.heatmap(dataframe1, vmin=0, vmax=1200, cmap='plasma', ax=ax1)
    # ax1.set_yticks(y_ticks)
    ax1.tick_params(bottom=False)
    ax2 = sns.heatmap(dataframe2, vmin=0, vmax=1200, cmap='plasma', ax=ax2)
    ax2.tick_params(bottom=False)
    ax3 = sns.heatmap(dataframe1-dataframe2, vmin=0, vmax=200, cmap='plasma', ax=ax3)
    plt.show()

def scatter_map(x, y, c):

    fig, ax = plt.subplots(1,1)

    z=c

    ax.scatter(x,y,z)
    ax.axline((0,0), (1000,1000), color='k', linestyle='--')
    ax.set_xlabel('Modellert innstråling (McClear)')
    ax.set_ylabel('Målt innstråling')
    plt.show()

def hist_plot_fordeling(ratio):
    ratio_bins = [round(x * 0.01, 2) for x in range(0, 201)]
    fig, ax = plt.subplots(1,1)
    ax.hist(ratio, bins=ratio_bins)

    ax.set_title('Histogram over fordeling av overirradiansmålinger')
    ax.set_xlabel('Forholdstall mellom målt og modellert innstråling')
    ax.set_ylabel('Frekvens')
    plt.show()

def hist_plot_varighet(lister):
    # Siden data er en "liste av lister" må dette flates ut.
    flat_list = [len for day in lister for len in day]
    print(flat_list)
    fig, ax = plt.subplots(1,1)
    ax.hist(flat_list, bins=100)

    ax.set_title('Fordeling av overirradiansepisoders varighet')
    ax.set_xlabel('Varighet av overirradiansepisoder (sekunder)')
    ax.set_ylabel('Frekvens')
    plt.show()


def snow_depth_scatter(idx, snow_depth, albedo):
#   Snødybdeplotting
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(idx, snow_depth, s=1, label='Snow_depth', color='blue')
    ax2.scatter(idx, albedo, s=1, label='Albedo', color='red')
    ax1.legend()

    plt.draw()
    plt.tight_layout()
    plt.show()

snow_depth_scatter(active_df.index, active_df['snow_depth'], active_df['albedo'])
# heat_map(merged_df['SWin'], merged_df['ghi_clear'], merged_df['time'], 'ten_seconds', len(unique_days))

# merged_df['time'] = pd.datetime(merged_df['time'], format='%H:%M:%S')
# merged_df['date'] = pd.to_datetime(merged_df['date'])

dataframe1 = active_df.pivot(index='time', columns='date', values='SWin')
dataframe2 = active_df.pivot(index='time', columns='date', values='ghi_clear')

def hist_varighet(oie_lengths):
    fig, ax = plt.subplots(1,1)

