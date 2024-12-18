import os
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import datetime as dt
import time
from itertools import groupby

# Timer
start_time = time.time()
# Angir startpunkt og sluttpunkt for data som ønskes behandlet
start = '2022-01-01 00:00:00'
end = '2022-12-31 23:59:50'

start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()
periode = f"{start_date} til {end_date}"

# Lager paths først
programmering_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
                     'v24\\Programmering\\'
mcclear_path = programmering_path + 'måledata/temp/mcclear_finse_2022.csv'
outpath = programmering_path + 'måledata/analyser/finse'

# Finner clear sky model for plots
raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename(columns={'Unnamed: 0': 'TIMESTAMP'}, inplace=True)
cs_full_df = raw_cs_df[['TIMESTAMP', 'ghi_extra', 'ghi_clear']]
cs_df = cs_full_df[(cs_full_df['TIMESTAMP'] >= start) & (cs_full_df['TIMESTAMP'] <= end)]

# Henter innstrålingsdata, her fra Finse
finse_22_file_path = programmering_path + '/måledata/Finse/Finseflux_level1_10sec_2022.csv'
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

# Calculates albedo, and sets NaN value for measurements not applicable
merged_df['albedo'] = merged_df['SWout'] / merged_df['SWin']
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


snow_presence_df = merged_df.groupby('date', group_keys=False).apply(calculate_snow_flag,
                                                                     include_groups=False).reset_index()

active_df = merged_df[merged_df['ghi_clear'] > 0]
active_df = active_df.copy()
active_df['oie'] = active_df['ghi_clear'] < active_df['SWin']


def get_max_measured_and_expected(group):
    if group['SWin'].notna().any():
        max_idx = group['SWin'].idxmax()
        max_value = group.loc[max_idx, 'SWin']
        expected_value = group.loc[max_idx, 'ghi_clear']
        return pd.Series([max_value, expected_value])
    else:
        return pd.Series([np.nan, np.nan])


max_active_df = active_df.groupby('date', group_keys=False).apply(get_max_measured_and_expected,
                                                                  include_groups=False).reset_index()
max_active_df.columns = ['date', 'max_value', 'expected_max_value']
max_active_df['max_ratio'] = max_active_df['max_value'] / max_active_df['expected_max_value']

simple_stats_active_df = active_df.groupby('date')['SWin'].agg(['median', 'mean', 'sum']).reset_index()
count_full_df = merged_df.groupby('date')['SWin'].count().reset_index()
count_full_df.columns = ['date', 'count']
stats_active_df = max_active_df.merge(simple_stats_active_df, on='date', how='left')
stats_active_df = stats_active_df.merge(count_full_df, on='date', how='left')
stats_active_df.columns = ['date', 'max_value', 'expected_max_value', 'max_ratio', 'median', 'mean', 'sum_measured',
                           'count']

stats_active_df['health'] = stats_active_df['count'] / (24 * 60 * 6)
stats_model_df = active_df.groupby('date')['ghi_clear'].sum().reset_index()
stats_model_df.columns = ['date', 'sum_model']

snow_avg_df = active_df.groupby('date')['snow_depth'].mean().reset_index()
albedo_avg_df = active_df.groupby('date')['albedo'].mean().reset_index()
snow_avg_df['snow_flag'] = snow_avg_df['snow_depth'] > 1


def calculate_oie_lengths(group):
    series = group['oie']
    oie_lengths = [sum(1 for _ in g) for k, g in groupby(series) if k]
    return pd.Series({'oie_lengths': oie_lengths})


oie_lengths_by_day = active_df.groupby('date', group_keys=False).apply(calculate_oie_lengths,
                                                                       include_groups=False).reset_index()
oie_lengths_by_day.columns = ['date', 'oie_lengths']

oie_count = active_df.groupby('date')['oie'].sum().reset_index()
oie_count.columns = ['date', 'oie_counts']

daily_stats_df = snow_avg_df.merge(oie_count, on='date', how='left')
daily_stats_df = daily_stats_df.merge(oie_lengths_by_day, on='date', how='left')
daily_stats_df = daily_stats_df.merge(stats_active_df, on='date', how='left')
daily_stats_df = daily_stats_df.merge(stats_model_df, on='date', how='left')
#daily_stats_df = daily_stats_df.merge(snow_avg_df, on='date', how='left')
daily_stats_df = daily_stats_df.merge(albedo_avg_df, on='date', how='left')
daily_stats_df['measured/expected'] = daily_stats_df['sum_measured'] / daily_stats_df['sum_model']
daily_stats_df['oie_counts'] = daily_stats_df['oie_counts'].fillna(0).astype(int)
daily_stats_df['num_oie'] = daily_stats_df['oie_lengths'].apply(len)


def heat_map(irr_data, cs_data, time, frequency, num_days):
    seconds = 86400
    ten_seconds = seconds // 10
    minutes = seconds // 60
    hours = seconds // 3600

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

    n = (time_of_day[0].hour * 3600 + time_of_day[0].minute * 60 + time_of_day[0].second) / f
    n = int(n)
    for i in range(num_days):
        for j in range(ten_seconds):
            # Check if there still is data in irr_data and cs_data
            if n < len(irr_data) and n < len(cs_data):
                irr_data_arr[j, i] = irr_data[n]
                cs_data_arr[j, i] = cs_data[n]
                n += 1
            else:
                # If dataset ends before assigned periods, break out
                break

    diff_data_arr = irr_data_arr - cs_data_arr

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1 = sns.heatmap(irr_data_arr, vmin=0, vmax=1200, cmap='plasma', ax=ax1)
    # ax1.set_ylim(0,24)
    # ax1.set_ylabel''
    #plt.draw()
    #plt.tight_layout()

    ax2 = sns.heatmap(cs_data_arr, vmin=0, vmax=1200, cmap='plasma', ax=ax2)

    ax3 = sns.heatmap(diff_data_arr, vmin=0, vmax=200, cmap='plasma', ax=ax3)

    plt.show()


def heat_map_improved(dataframe1, dataframe2):
    y_labels = ['01:00:00', '03:00:00', '05:00:00', '07:00:00', '09:00:00', '11:00:00',
                '13:00:00', '15:00:00', '17:00:00', '19:00:00', '21:00:00', '23:00:00']
    fig, (ax1, ax3) = plt.subplots(2, 1)
    ax1 = sns.heatmap(dataframe1, vmin=0, vmax=1500, cmap='plasma', ax=ax1)
    # ax1.set_yticks(y_ticks)
    ax1.tick_params(bottom=False, labelbottom=False)
    # ax2 = sns.heatmap(dataframe2, vmin=0, vmax=1200, cmap='plasma', ax=ax2)
    # ax2.tick_params(bottom=False, labelbottom=False)
    ax3 = sns.heatmap(dataframe1 - dataframe2, vmin=0, vmax=600, cmap='plasma', ax=ax3)

    ax1.set_yticklabels(y_labels)
    ax1.set_ylabel('Tid')

    # ax2.set_yticklabels(y_labels)
    # ax2.set_ylabel('Tid')

    ax3.set_yticklabels(y_labels)
    ax3.set_ylabel('Tid')

    #ax3.set_xticklabels(rotation=45)
    ax3.set_xlabel('Dato')
    plt.tight_layout()
    plt.show()


def scatter_map(x, y, c):
    fig, ax = plt.subplots(1, 1)

    z = c

    ax.scatter(x, y, z)
    ax.axline((0, 0), (1000, 1000), color='k', linestyle='--')
    ax.set_xlabel('Modellert innstråling (McClear)')
    ax.set_ylabel('Målt innstråling')
    plt.show()


def hist_plot_fordeling(ratio):
    ratio_bins = [round(x * 0.01, 2) for x in range(0, 201)]
    fig, ax = plt.subplots(1, 1)
    ax.hist(ratio, bins=ratio_bins)

    ax.set_title(f'Histogram over fordeling av overirradiansmålinger \ni perioden {periode}')
    ax.set_xlabel('Forholdstall mellom målt og modellert innstråling')
    ax.set_ylabel('Frekvens')
    plt.show()


def hist_plot_varighet(lister):
    # Siden data er en "liste av lister" må dette flates ut.
    flat_list = [len for day in lister for len in day]
    bin_width = 6  # antall 10-sekunders bokser
    max_value = int(np.ceil(np.max(flat_list)))
    bins = np.arange(0, max_value + bin_width, bin_width)

    fig, ax = plt.subplots(1, 1)
    ax.hist(flat_list, bins=bins)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_axisbelow(True)
    ax.grid(True, which='both', ls='-', color='0.85')
    ax.set_title(f'Fordeling av overirradiansepisoders varighet \ni perioden {periode}')
    ax.set_xlabel('Antall sammenhengende overirradiansmålinger')
    ax.set_ylabel('Frekvens')
    plt.show()


def snow_depth_scatter(idx, snow_depth, albedo):
    #   Snødybdeplotting med albedo
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(idx, snow_depth, s=1, label='Snødybde', color='blue')
    ax2.scatter(idx, albedo, s=1, label='Albedo', color='red')
    ax1.tick_params(bottom=False)
    ax1.set_ylabel('Snødybde (mm)')
    ax2.set_ylabel('Albedo')
    ax2.set_xlabel('Dato')
    fig.suptitle(f'Gjennomsnittlig snødybde og albedo \nfor perioden {periode}')
    fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.85))

    plt.draw()
    plt.tight_layout()
    plt.show()


snow_depth_scatter(daily_stats_df['date'], daily_stats_df['snow_depth'], daily_stats_df['albedo'])
# heat_map(merged_df['SWin'], merged_df['ghi_clear'], merged_df['time'], 'ten_seconds', len(unique_days))

# merged_df['time'] = pd.datetime(merged_df['time'], format='%H:%M:%S')
# merged_df['date'] = pd.to_datetime(merged_df['date'])

dataframe1 = active_df.pivot(index='time', columns='date', values='SWin')
dataframe2 = active_df.pivot(index='time', columns='date', values='ghi_clear')

heat_map_improved(dataframe1, dataframe2)
hist_plot_fordeling(active_df['SWin'] / active_df['ghi_clear'])
hist_plot_varighet(oie_lengths_by_day['oie_lengths'])
#scatter_map(active_df['ghi_clear'], active_df['SWin'], active_df['SWin']/active_df['ghi_clear'])


active_df['oie_intensity'] = (active_df['SWin'] - active_df['ghi_clear']).clip(lower=0)
active_df.loc[:, 'oie_flag'] = active_df['ghi_clear'] < active_df['SWin']

active_df['event_id'] = (
        (active_df['oie_flag'] != active_df['oie_flag'].shift()) |
        (active_df['date'] != active_df['date'].shift())
).cumsum()
active_df['event_id'] = active_df['event_id'].where(active_df['oie_flag'], np.nan)

events_finse = (
    active_df.dropna(subset=['event_id'])
    .groupby('event_id')
    .agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        duration=('time', lambda x: int(((x.index.max() - x.index.min()).total_seconds() + 10))),
        total_intensity=('oie_intensity', 'sum'),
        avg_intensity=('oie_intensity', 'mean'),
        max_intensity=('oie_intensity', 'max'),
        date=('date', 'first'),
    )
)

duration_bins = [0, 10, 60, 3600, np.inf]
duration_labels = ['2', '3', '4', '5']
events_finse['duration_category'] = pd.cut(events_finse['duration'], bins=duration_bins, labels=duration_labels,
                                           right=True, include_lowest=True)

intensity_bins_1h = [0, 10, 50, 100, 200, np.inf]
intensity_labels_1h = ['A', 'B', 'C', 'D', 'E']
events_finse['intensity_category'] = pd.cut(events_finse['total_intensity'], bins=intensity_bins_1h,
                                            labels=intensity_labels_1h, right=False)

events_finse['combined_category'] = events_finse['duration_category'].astype(str) + " & " + events_finse[
    'intensity_category'].astype(str)
category_counts_10s = events_finse['combined_category'].value_counts()

combined_category_order_10s = [
    "2 & A", "2 & B", "2 & C", "2 & D", "2 & E",
    "3 & A", "3 & B", "3 & C", "3 & D", "3 & E",
    "4 & A", "4 & B", "4 & C", "4 & D", "4 & E",
    "5 & A", "5 & B", "5 & C", "5 & D", "5 & E"
]
category_counts_10s_ordered = category_counts_10s.reindex(combined_category_order_10s, fill_value=0)

axh = category_counts_10s_ordered.plot(kind='bar', figsize=(10, 6),
                                       title=f"Kategorisk inndeling av overirradiansepisoder \nbasert på irradiansdata fra Finse fra {periode}")
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=60)
plt.tight_layout()

for bar in axh.patches:
    yval = bar.get_height()
    axh.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha='center',
        va='bottom')

plt.show()

valued_df_resampled_1m = active_df[['ghi_clear', 'SWin']].resample('60s').mean()
valued_df_resampled_1m['date'] = valued_df_resampled_1m.index.date
valued_df_resampled_1m['time'] = valued_df_resampled_1m.index.time
valued_df_resampled_1m['oie_intensity'] = (valued_df_resampled_1m['SWin'] - valued_df_resampled_1m['ghi_clear']).clip(
    lower=0)
valued_df_resampled_1m.loc[:, 'oie_flag'] = valued_df_resampled_1m['ghi_clear'] < valued_df_resampled_1m['SWin']

valued_df_resampled_1m['event_id'] = (
        (valued_df_resampled_1m['oie_flag'] != valued_df_resampled_1m['oie_flag'].shift()) |
        (valued_df_resampled_1m['date'] != valued_df_resampled_1m['date'].shift())
).cumsum()
valued_df_resampled_1m['event_id'] = valued_df_resampled_1m['event_id'].where(valued_df_resampled_1m['oie_flag'],
                                                                              np.nan)

events_1m = (
    valued_df_resampled_1m.dropna(subset=['event_id'])
    .groupby('event_id')
    .agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        duration=('time', lambda x: int(((x.index.max() - x.index.min()).total_seconds()) / 60 + 1)),
        # assumes timestamp is datetime
        total_intensity=('oie_intensity', 'sum'),
        avg_intensity=('oie_intensity', 'mean'),
        max_intensity=('oie_intensity', 'max'),
        date=('date', 'first'),
    )
)

duration_bins_1m = [0, 1, 60, np.inf]
duration_labels_1m = ['3', '4', '5']
events_1m['duration_category'] = pd.cut(events_1m['duration'], bins=duration_bins_1m, labels=duration_labels_1m,
                                        right=True, include_lowest=True)

intensity_bins_1m = [0, 10, 50, 100, 200, np.inf]
intensity_labels_1m = ['A', 'B', 'C', 'D', 'E']
events_1m['intensity_category'] = pd.cut(events_1m['total_intensity'], bins=intensity_bins_1m,
                                         labels=intensity_labels_1m, right=False)

events_1m['combined_category'] = events_1m['duration_category'].astype(str) + " & " + events_1m[
    'intensity_category'].astype(str)
category_counts_1m = events_1m['combined_category'].value_counts()

combined_category_order_1m = [
    "3 & A", "3 & B", "3 & C", "3 & D", "3 & E",
    "4 & A", "4 & B", "4 & C", "4 & D", "4 & E",
    "5 & A", "5 & B", "5 & C", "5 & D", "5 & E"
]
category_counts_1m_ordered = category_counts_1m.reindex(combined_category_order_1m, fill_value=0)

axm = category_counts_1m_ordered.plot(kind='bar', figsize=(10, 6),
                                      title=f"Kategorisk inndeling av overirradiansepisoder \nbasert på minuttmidlet 10-sekunddata fra Finse fra {periode}")
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=60)
plt.tight_layout()
for bar in axm.patches:
    yval = bar.get_height()
    axm.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha='center',
        va='bottom')

plt.show()

valued_df_resampled_1h = active_df[['ghi_clear', 'SWin']].resample('1h').mean()
valued_df_resampled_1h['date'] = valued_df_resampled_1h.index.date
valued_df_resampled_1h['time'] = valued_df_resampled_1h.index.time
valued_df_resampled_1h['oie_intensity'] = (valued_df_resampled_1h['SWin'] - valued_df_resampled_1h['ghi_clear']).clip(
    lower=0)
valued_df_resampled_1h.loc[:, 'oie_flag'] = valued_df_resampled_1h['ghi_clear'] < valued_df_resampled_1h['SWin']

valued_df_resampled_1h['event_id'] = (
        (valued_df_resampled_1h['oie_flag'] != valued_df_resampled_1h['oie_flag'].shift()) |
        (valued_df_resampled_1h['date'] != valued_df_resampled_1h['date'].shift())
).cumsum()

valued_df_resampled_1h['event_id'] = valued_df_resampled_1h['event_id'].where(valued_df_resampled_1h['oie_flag'],
                                                                              np.nan)

events_1h = (
    valued_df_resampled_1h.dropna(subset=['event_id'])
    .groupby('event_id')
    .agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        duration=('time', lambda x: int(((x.index.max() - x.index.min()).total_seconds()) / 3600 + 1)),
        total_intensity=('oie_intensity', 'sum'),
        avg_intensity=('oie_intensity', 'mean'),
        max_intensity=('oie_intensity', 'max'),
        date=('date', 'first'),
    )
)

duration_bins_1h = [0, 1, np.inf]
duration_labels_1h = ['4', '5']
events_1h['duration_category'] = pd.cut(events_1h['duration'],
                                        bins=duration_bins_1h,
                                        labels=duration_labels_1h,
                                        right=True)

intensity_bins_1h = [0, 10, 50, 100, 200, np.inf]
intensity_labels_1h = ['A', 'B', 'C', 'D', 'E']
events_1h['intensity_category'] = pd.cut(events_1h['total_intensity'],
                                         bins=intensity_bins_1h,
                                         labels=intensity_labels_1h,
                                         right=False)

events_1h['combined_category'] = events_1h['duration_category'].astype(str) + " & " + events_1h[
    'intensity_category'].astype(str)
category_counts_1h = events_1h['combined_category'].value_counts()

combined_category_order_1h = [
    "4 & A", "4 & B", "4 & C", "4 & D", "4 & E",
    "5 & A", "5 & B", "5 & C", "5 & D", "5 & E"
]
category_counts_1h_ordered = category_counts_1h.reindex(combined_category_order_1h, fill_value=0)

axh = category_counts_1h_ordered.plot(kind='bar', figsize=(10, 6),
                                      title=f"Kategorisk inndeling av overirradiansepisoder \nbasert på timesmidlet 10-sekunddata fra Finse fra {periode}")
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=60)
plt.tight_layout()

for bar in axh.patches:
    yval = bar.get_height()
    axh.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha='center',
        va='bottom')

plt.show()


events_finse = events_finse.merge(
    snow_avg_df,
    left_on='date',
    right_on='date',
    how='left'
)

category_snow_counts = (
    events_finse.groupby(['combined_category', 'snow_flag'])
    .size()
    .reset_index(name='count')
    .pivot(index='combined_category', columns='snow_flag', values='count')
    .fillna(0)
    .astype(int)
)

category_snow_counts.columns = ['No Snow', 'Snow']
category_snow_counts = category_snow_counts.rename_axis("Kategorier")

ax = category_snow_counts.plot(
    kind='bar',
    stacked=False,
    figsize=(10, 6),
    color=['steelblue', 'skyblue'],  # Colors for No Snow and Snow
    title=f"Overirradiansepisoder på Finse kategorisert og sortert etter snøforhold\ni periode {periode}",
)

# Add labels
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=45, ha='right')  # Rotate xticks for better readability
plt.legend(title="Snøforhold", labels=["Ingen snø", "Snø"])
plt.ylim(0,2300)
plt.tight_layout()


for bar in ax.patches:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha='center',
        va='bottom',
        fontsize=10,
        color='black'
    )

plt.show()


selected_dates = ['2023-05-02', '2023-05-07', '2023-05-09']

df_filtered = daily_stats_df[daily_stats_df['date'].isin(selected_dates)]
df_to_latex = df_filtered[['date', 'max_value', 'expected_max_value', 'max_ratio', 'oie_counts', 'num_oie', 'snow_flag']]

df_to_latex = df_to_latex.rename(columns={
    'max_value': 'Målt maksverdi',
    'expected_max_value': 'Forventet verdi',
    'max_ratio': 'Forholdstall',
    'oie_counts': 'Antall målinger over forventet',
    'num_oie': 'Antall sammenhengende OIE',
    'snow_flag': 'Snøforhold'
})

# Display the resulting DataFrame
print(df_to_latex)

# Convert the DataFrame to LaTeX format
latex_code = df_to_latex.to_latex(index=False)

# Print the LaTeX code
print(latex_code)