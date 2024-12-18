# import pandas as pd
# import numpy as np
# import os
# import pickle
# from datetime import datetime, timedelta
# from matplotlib import pyplot as plt
# import time
#
# # Setter tidspunkt for datasettet
# start = "2023-08-09 00:00:00"
# end = "2023-08-10 00:00:00"
# start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
# end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()
#
# # Må skille på om data vil hentes fra tak eller fra solpark når data hentes - 'Roof' og 'Solpark'
# data_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
#             'v24\\Programmering\\måledata'
# mcclear_path = data_path + '\\temp\\mcclear_søråsjordet_2023.csv'
# outpath = data_path + '\\plots\\kjeller'
#
# with open(data_path + '\\kjeller\\Solpark\\KZPR_IRHoSunP_20210820_20230925.pkl', 'rb') as pickle_in:
#     solar_sunh_data_raw = pickle.load(pickle_in)
# solar_sunh_data = solar_sunh_data_raw[start:end].to_frame('sol_h')
#
# raw_cs_df = pd.read_csv(mcclear_path)
# raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
# raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
# raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
# raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=1)
# raw_cs_df.set_index('TIMESTAMP', inplace=True)
# old_df = raw_cs_df.loc[start_date:end_date]
# cs_df = old_df.drop('Observation period', axis=1)
#
# # Fyller inn cs-modellen
# cs_df.astype(float)
# cs_filled_df = cs_df.reindex(pd.date_range(start=start_date, end=end, freq='s'))
# cs_filled_df.index = pd.to_datetime(cs_filled_df.index)
# cs_filled_df.interpolate(method='polynomial', order=2, inplace=True)
#
# valued_df = pd.merge(cs_filled_df, solar_sunh_data, left_index=True, right_index=True, how='outer')
# valued_df = valued_df[valued_df.index.date == start_date]
# unique_time = [f"{hour:02d}:{minute:02d}:{second:02d}"
#                        for hour in range(24)
#                        for minute in range(60)
#                        for second in range(60)]
# valued_df.loc[:, 'time'] = unique_time
#
# valued_df['10s_smoothed'] = valued_df['sol_h'].rolling(window=10, center = True).mean()
# valued_df['minute_smoothed'] = valued_df['sol_h'].rolling(window=60, center = True).mean()
# # valued_df = merged_df[merged_df['ghi_extra'] > 1]
# test_df = valued_df[(valued_df.index.hour >= 15) &
#                     (valued_df.index.hour < 18)]
#
# fig, ax = plt.subplots()
# ghi, = ax.plot(test_df['time'], test_df['ghi_clear'], label='GHI')
# toa, = ax.plot(test_df['time'], test_df['ghi_extra'], label='TOA')
# sol_h = ax.scatter(test_df['time'], test_df['sol_h'], s=1, label='sol_h')
# sol_h_10s = ax.scatter(test_df['time'], test_df['10s_smoothed'], s=1, label='10s')
# sol_h_1m = ax.scatter(test_df['time'], test_df['minute_smoothed'], s=1, label='1m')
# plt.xticks(test_df['time'][0::3600], rotation=70)
# ax.legend(loc='upper right')
# # plt.draw()
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import time
from itertools import groupby
from zoneinfo import ZoneInfo

# Setter tidspunkt for datasettet
start = "2022-10-01 00:00:00"
end = "2023-09-30 23:59:59"
start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()

periode = f"{start_date} til {end_date}"

summer_start = pd.Timestamp('2023-03-26 02:00:00')  # Sommertid start mars
summer_end = pd.Timestamp('2022-10-30 03:00:00')    # Sommertid slutt oktober

# Må skille på om data vil hentes fra tak eller fra solpark når data hentes - 'Roof' og 'Solpark'
data_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata'
mcclear_path = data_path + '\\temp\\mcclear_kjeller_2021_2023.csv'
outpath = data_path + '\\plots\\kjeller'

with open(data_path + '\\kjeller\\Roof\\OEhor_irr_20210101_20231231.pkl', 'rb') as pickle_in:
    roof_ref_data_raw = pickle.load(pickle_in)
roof_ref_data_raw.set_index('date_time', inplace=True)
roof_ref_data_series = roof_ref_data_raw['OEhor_irr']
roof_ref_data = roof_ref_data_series[start:end].to_frame('roof_ref')

raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)

# Takler sommertid
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'] + pd.Timedelta(hours=2)
raw_cs_df.loc[(raw_cs_df['TIMESTAMP'] >= summer_end) & (raw_cs_df['TIMESTAMP'] < summer_start), 'TIMESTAMP'] -= pd.Timedelta(hours=1)
raw_cs_df = raw_cs_df[~raw_cs_df['TIMESTAMP'].duplicated(keep='first')]

raw_cs_df.set_index('TIMESTAMP', inplace=True)
columns_to_drop = ['Observation period', 'bhi_clear', 'dni_clear', 'dhi_clear']
cs_df = raw_cs_df.drop(columns_to_drop, axis=1).loc[start_date:end_date]

# Fyller inn cs-modellen
cs_df.astype(float)
cs_filled_df = cs_df.reindex(pd.date_range(start=start_date, end=end, freq='s'))
cs_filled_df.index = pd.to_datetime(cs_filled_df.index)
cs_filled_df.interpolate(method='polynomial', order=2, inplace=True)

pre_valued_df = pd.merge(cs_filled_df, roof_ref_data, left_index=True, right_index=True, how='outer')
#dated_valued_df = pre_valued_df[pre_valued_df.index.date == start_date]
#pre_valued_df.loc[:,'date'] = pre_valued_df.index.date
dated_valued_df = pre_valued_df.loc[start_date:end_date]

dated_valued_df.loc[:,'date'] = dated_valued_df.index.date
unique_time = [f"{hour:02d}:{minute:02d}:{second:02d}"
                       for hour in range(24)
                       for minute in range(60)
                       for second in range(60)]
# Må tas med når det plottes...
# dated_valued_df.loc[:, 'time'] = unique_time
valued_df = dated_valued_df[dated_valued_df['ghi_clear']>50]
valued_df['time'] = valued_df.index.time


# valued_df['10s_smoothed'] = valued_df['sol_h'].rolling(window=10, center = True).mean()
# valued_df['minute_smoothed'] = valued_df['sol_h'].rolling(window=60, center = True).mean()


def calculate_corr_coeff(df):
    results = []

    for date, group in df.groupby('date'):
        measured = group['roof_ref']
        model = group['ghi_clear']

        valid_data = pd.concat([measured, model], axis=1).dropna()
        if len(valid_data) > 1:
            r = np.corrcoef(valid_data['roof_ref'], valid_data['ghi_clear'])[0, 1]
        else:
            r = np.nan
        results.append({'date': date, 'r': r})

    corr_coeff_df = pd.DataFrame(results)
    return corr_coeff_df

corr_coeff_df = calculate_corr_coeff(valued_df)

corr_coeff_df['date'] = corr_coeff_df['date'].apply(
    lambda x: x if isinstance(x, tuple) else x)
# corr_coeff_df['date'] = pd.to_datetime(corr_coeff_df['date'])
# valued_df['real_date'] = pd.to_datetime(valued_df['date'])

def calculate_correction_factor(corr_coeff_df, df):
    # Setter grense på 0.99 for samsvar i data for beregning av korreksjonsfaktor
    clear_days = corr_coeff_df[corr_coeff_df['r'] > 0.99]

    total_ratio = 0
    count = 0

    for date in clear_days['date']:
        day_data = df[df['date'] == date]
        measured = day_data['roof_ref']
        modeled = day_data['ghi_clear']

        # Calculate the ratio of measured to modeled for this day (ignoring NaN values)
        valid_data = pd.concat([measured, modeled], axis=1).dropna()
        if len(valid_data) > 0:
            ratio = valid_data['roof_ref'].sum() / valid_data['ghi_clear'].sum()
            total_ratio += ratio
            count += 1

    correction_factor = total_ratio/count
    return correction_factor

# correction_factor = calculate_correction_factor(corr_coeff_df, valued_df)
#print(f"Korreksjonsfaktoren er {correction_factor}")


valued_df.loc[:,'oie_intensity'] = (valued_df['roof_ref']-valued_df['ghi_clear']).clip(lower=0)
valued_df.loc[:,'oie_flag'] = valued_df['ghi_clear'] < valued_df['roof_ref']

valued_df['event_id'] = (valued_df['oie_flag'] != valued_df['oie_flag'].shift()).cumsum()
valued_df['event_id'] = valued_df['event_id'].where(valued_df['oie_flag'], np.nan)

events = (
    valued_df.dropna(subset=['event_id'])
    .groupby('event_id')
    .agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        duration=('time', lambda x: int((x.index.max() - x.index.min()).total_seconds()+1)),  # assumes timestamp is datetime
        total_intensity=('oie_intensity', 'sum'),
        avg_intensity=('oie_intensity', 'mean'),
        max_intensity=('oie_intensity', 'max'),
        date = ('date', 'first'),
    )
)

duration_bins = [0, 1, 10, 60, 3600, np.inf]
duration_labels = ['1', '2', '3', '4', '5']
events['duration_category'] = pd.cut(events['duration'], bins=duration_bins, labels=duration_labels, right=True, include_lowest=True)

intensity_bins = [0, 10, 50, 100, 200, np.inf]
intensity_labels = ['A', 'B', 'C', 'D', 'E']
events['intensity_category'] = pd.cut(events['total_intensity'], bins=intensity_bins, labels=intensity_labels, right=False)

events['combined_category'] = events['duration_category'].astype(str) + " & " + events['intensity_category'].astype(str)
category_counts_1s = events['combined_category'].value_counts()

combined_category_order_1s = [
    "1 & A", "1 & B", "1 & C", "1 & D", "1 & E",
    "2 & A", "2 & B", "2 & C", "2 & D", "2 & E",
    "3 & A", "3 & B", "3 & C", "3 & D", "3 & E",
    "4 & A", "4 & B", "4 & C", "4 & D", "4 & E",
    "5 & A", "5 & B", "5 & C", "5 & D", "5 & E"
]
category_counts_1s_ordered = category_counts_1s.reindex(combined_category_order_1s, fill_value=0)


axs = category_counts_1s_ordered.plot(kind='bar', figsize=(10, 6), title=f"Kategorisk inndeling av overirradiansepisoder \nbasert på sekunddata fra Kjeller fra {periode}")
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=60)
plt.tight_layout()
for bar in axs.patches:
    yval = bar.get_height()
    axs.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.show()


valued_df_resampled_1m = valued_df[['ghi_clear', 'roof_ref']].resample('60s').mean()
valued_df_resampled_1m['date'] = valued_df_resampled_1m.index.date
valued_df_resampled_1m['time'] = valued_df_resampled_1m.index.time
valued_df_resampled_1m['oie_intensity'] = (valued_df_resampled_1m['roof_ref']-valued_df_resampled_1m['ghi_clear']).clip(lower=0)
valued_df_resampled_1m.loc[:,'oie_flag'] = valued_df_resampled_1m['ghi_clear'] < valued_df_resampled_1m['roof_ref']

valued_df_resampled_1m['event_id'] = (valued_df_resampled_1m['oie_flag'] != valued_df_resampled_1m['oie_flag'].shift()).cumsum()
valued_df_resampled_1m['event_id'] = valued_df_resampled_1m['event_id'].where(valued_df_resampled_1m['oie_flag'], np.nan)

events_1m = (
    valued_df_resampled_1m.dropna(subset=['event_id'])
    .groupby('event_id')
    .agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        duration=('time', lambda x: int(((x.index.max() - x.index.min()).total_seconds())/60+1)),  # assumes timestamp is datetime
        total_intensity=('oie_intensity', 'sum'),
        avg_intensity=('oie_intensity', 'mean'),
        max_intensity=('oie_intensity', 'max'),
        date=('date', 'first'),
    )
)

duration_bins_1m = [0, 1, 60, np.inf]
duration_labels_1m = ['3', '4', '5']
events_1m['duration_category'] = pd.cut(events_1m['duration'], bins=duration_bins_1m, labels=duration_labels_1m, right=True, include_lowest=True)

intensity_bins_1m = [0, 10, 50, 100, 200, np.inf]
intensity_labels_1m = ['A', 'B', 'C', 'D', 'E']
events_1m['intensity_category'] = pd.cut(events_1m['total_intensity'], bins=intensity_bins_1m, labels=intensity_labels_1m, right=False)

events_1m['combined_category'] = events_1m['duration_category'].astype(str) + " & " + events_1m['intensity_category'].astype(str)
category_counts_1m = events_1m['combined_category'].value_counts()

combined_category_order_1m = [
    "3 & A", "3 & B", "3 & C", "3 & D", "3 & E",
    "4 & A", "4 & B", "4 & C", "4 & D", "4 & E",
    "5 & A", "5 & B", "5 & C", "5 & D", "5 & E"
]
category_counts_1m_ordered = category_counts_1m.reindex(combined_category_order_1m, fill_value=0)


axm = category_counts_1m_ordered.plot(kind='bar', figsize=(10, 6), title=f"Kategorisk inndeling av overirradiansepisoder \nbasert på minuttmidlet sekunddata fra Kjeller fra {periode}")
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=60)
plt.tight_layout()
for bar in axm.patches:
    yval = bar.get_height()
    axm.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()


valued_df_resampled_1h = valued_df[['ghi_clear', 'roof_ref']].resample('1h').mean()
valued_df_resampled_1h['date'] = valued_df_resampled_1h.index.date
valued_df_resampled_1h['time'] = valued_df_resampled_1h.index.time
valued_df_resampled_1h['oie_intensity'] = (valued_df_resampled_1h['roof_ref']-valued_df_resampled_1h['ghi_clear']).clip(lower=0)
valued_df_resampled_1h.loc[:,'oie_flag'] = valued_df_resampled_1h['ghi_clear'] < valued_df_resampled_1h['roof_ref']

valued_df_resampled_1h['event_id'] = (valued_df_resampled_1h['oie_flag'] != valued_df_resampled_1h['oie_flag'].shift()).cumsum()
valued_df_resampled_1h['event_id'] = valued_df_resampled_1h['event_id'].where(valued_df_resampled_1h['oie_flag'], np.nan)

events_1h = (
    valued_df_resampled_1h.dropna(subset=['event_id'])
    .groupby('event_id')
    .agg(
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        duration=('time', lambda x: int(((x.index.max() - x.index.min()).total_seconds())/3600+1)),
        total_intensity=('oie_intensity', 'sum'),
        avg_intensity=('oie_intensity', 'mean'),
        max_intensity=('oie_intensity', 'max'),
        date=('date', 'first'),
    )
)

duration_bins_1h = [0, 1, np.inf]
duration_labels_1h = ['4', '5']
events_1h['duration_category'] = pd.cut(events_1h['duration'], bins=duration_bins_1h, labels=duration_labels_1h, right=True)

intensity_bins_1h = [0, 10, 50, 100, 200, np.inf]
intensity_labels_1h = ['A', 'B', 'C', 'D', 'E']
events_1h['intensity_category'] = pd.cut(events_1h['total_intensity'], bins=intensity_bins_1h, labels=intensity_labels_1h, right=False)

events_1h['combined_category'] = events_1h['duration_category'].astype(str) + " & " + events_1h['intensity_category'].astype(str)
category_counts_1h = events_1h['combined_category'].value_counts()

combined_category_order_1h = [
    "4 & A", "4 & B", "4 & C", "4 & D", "4 & E",
    "5 & A", "5 & B", "5 & C", "5 & D", "5 & E"
]
category_counts_1h_ordered = category_counts_1h.reindex(combined_category_order_1h, fill_value=0)

axh = category_counts_1h_ordered.plot(kind='bar', figsize=(10, 6), title=f"Kategorisk inndeling av overirradiansepisoder \nbasert på timesmidlet sekunddata fra Kjeller fra {periode}")
plt.xlabel("Kombinerte kategorier")
plt.ylabel("Antall overirradiansepisoder")
plt.xticks(rotation=60)
plt.tight_layout()

for bar in axh.patches:
    yval = bar.get_height()
    axh.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()


# ANY WAY TO GROUP BY MONTH?

def calculate_oie_lengths (group):
    series = group['oie_flag']
    oie_lengths = [sum(1 for _ in g) for k, g in groupby(series) if k]
    return pd.Series({'oie_lengths': oie_lengths})

# 1s
oie_lengths_by_day = valued_df.groupby('date', group_keys=False).apply(calculate_oie_lengths, include_groups=False).reset_index()
oie_lengths_by_day.columns = ['date', 'oie_lengths']
oie_count = valued_df.groupby('date')['oie_flag'].sum().reset_index()
oie_count.columns = ['date', 'oie_counts']

#1m
oie_lengths_by_day_1m = valued_df_resampled_1m.groupby('date', group_keys=False).apply(calculate_oie_lengths, include_groups=False).reset_index()
oie_lengths_by_day_1m.columns = ['date', 'oie_lengths']
oie_count_1m = valued_df_resampled_1m.groupby('date')['oie_flag'].sum().reset_index()
oie_count_1m.columns = ['date', 'oie_counts']

#1h
oie_lengths_by_day_1h = valued_df_resampled_1h.groupby('date', group_keys=False).apply(calculate_oie_lengths, include_groups=False).reset_index()
oie_lengths_by_day_1h.columns = ['date', 'oie_lengths']
oie_count_1h = valued_df_resampled_1h.groupby('date')['oie_flag'].sum().reset_index()
oie_count_1h.columns = ['date', 'oie_counts']


# valued_df.plot(y=['ghi_clear', 'roof_ref'],
#                title='Sekunddata',
#                label=['GHI_ClearSky', 'Måledata'])
# plt.xlabel('Tidspunkt (hh:mm)')
# plt.ylabel('Irradians (Wm^-2)')
# plt.ylim(0,900)
# plt.show()
# valued_df_resampled_1m.plot(y=['ghi_clear', 'roof_ref'],
#                             title='Minuttdata midlet fra sekunddata',
#                             label=['GHI_ClearSky', 'Måledata']
#                             )
# plt.ylim(0,900)
# plt.xlabel('Tidspunkt (hh:mm)')
# plt.ylabel('Irradians (Wm^-2)')
# plt.show()
# valued_df_resampled_1h.plot(y=['ghi_clear', 'roof_ref'],
#                             title='Timesdata midlet fra sekunddata',
#                             label=['GHI_ClearSky', 'Måledata']
#                             )
# plt.ylim(0,900)
# plt.xlabel('Tidspunkt (hh:mm)')
# plt.ylabel('Irradians (Wm^-2)')
# plt.show()
#


def hist_plot_varighet(lister, bin_width):

    # Siden data er en "liste av lister" må dette flates ut.
    flat_list = [len for day in lister for len in day]
    if len(flat_list) == 0:
        print('Tom flat liste')
        return

    max_value = int(np.ceil(np.max(flat_list)))
    bins = np.arange(0, max_value + bin_width, bin_width)

    fig, ax = plt.subplots(1,1)
    ax.hist(flat_list, bins=bins)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_axisbelow(True)
    ax.grid(True, which='both', ls='-', color='0.85')
    ax.set_title('Fordeling av overirradiansepisoders varighet')
    ax.set_xlabel('Lengde på sammenhengende overirradiansmålinger (s)')
    ax.set_ylabel('Frekvens')
    plt.show()


def analyze_irradiance(dataframe):
    total_episodes = 0
    episodes_1_tick = 0
    episodes_shorter_10_s = 0
    episodes_shorter_minute_full = 0
    episodes_shorter_minute_sec = 0
    episodes_shorter_hour = 0
    episodes_longer_hour = 0


    for episode_list in dataframe['oie_lengths']:
        total_episodes += len(episode_list)

        episodes_1_tick +=          sum(1 for x in episode_list if x == 1)
        episodes_shorter_10_s +=    sum(1 for x in episode_list if 1 < x <= 10)
        episodes_shorter_minute_full += sum(1 for x in episode_list if 1 < x <= 60)
        episodes_shorter_minute_sec +=  sum(1 for x in episode_list if 10 < x <= 60)
        episodes_shorter_hour +=    sum(1 for x in episode_list if 60 < x <= 3600)
        episodes_longer_hour +=     sum(1 for x in episode_list if x > 3600)

    result = {
        "Total episodes": total_episodes,
        "Episodes 1 tick long": episodes_1_tick,
        "Episodes shorter than 10 ticks": episodes_shorter_10_s,
        "Episodes shorter than 60 ticks 1-60": episodes_shorter_minute_full,
        "Episodes shorter than 60 ticks 10-60": episodes_shorter_minute_sec,
        "Episodes shorter than 3600 ticks": episodes_shorter_hour,
        "Episodes longer than 3600 ticks": episodes_longer_hour,
    }
    return result

result_1s = analyze_irradiance(oie_lengths_by_day)
result_1m = analyze_irradiance(oie_lengths_by_day_1m)
result_1h = analyze_irradiance(oie_lengths_by_day_1h)



all_numbers = range(1, 6)  # For duration categories: 1-5
all_letters = ['A', 'B', 'C', 'D', 'E']  # For intensity categories: a-e


def print_occurrences(duration_counts, intensity_counts, label):
    duration_counts = duration_counts.reindex(all_numbers, fill_value=0)
    intensity_counts = intensity_counts.reindex(all_letters, fill_value=0)

    print(f"Occurrences of Duration and Intensity categories, {label}:")
    for number, n_count in duration_counts.items():
        letter = all_letters[number - 1]
        l_count = intensity_counts[letter]
        print(
            f"There are {n_count} occurrences of the number {number} and {l_count} occurrences of the letter '{letter}'.")


# First dataset (1s)
duration_counts = events['duration_category'].value_counts().sort_index()
intensity_counts = events['intensity_category'].value_counts().sort_index()
print_occurrences(duration_counts, intensity_counts, "1s")

# Second dataset (1m)
duration_counts_m = events_1m['duration_category'].value_counts().sort_index()
intensity_counts_m = events_1m['intensity_category'].value_counts().sort_index()
print_occurrences(duration_counts_m, intensity_counts_m, "1m")

# Third dataset (1h)
duration_counts_h = events_1h['duration_category'].value_counts().sort_index()
intensity_counts_h = events_1h['intensity_category'].value_counts().sort_index()
print_occurrences(duration_counts_h, intensity_counts_h, "1h")

print(f" S Sum of durations: {duration_counts.sum()} and intensity: {intensity_counts.sum()}")
print(f" M Sum of durations: {duration_counts_m.sum()} and intensity: {intensity_counts_m.sum()}")
print(f" H Sum of durations: {duration_counts_h.sum()} and intensity: {intensity_counts_h.sum()}")

def hist_plot_fordeling(ratio, type):
    ratio_bins = [round(x * 0.01, 2) for x in range(0, 201)]
    fig, ax = plt.subplots(1,1)
    ax.hist(ratio, bins=ratio_bins)

    ax.set_title(f"Fordeling av overirradiansmålingers forholdstall basert på \n {type} på Kjeller fra {periode}")
    ax.set_xlabel('Forholdstall mellom målt og modellert innstråling')
    ax.set_ylabel('Antall målinger')
    plt.tight_layout()
    plt.show()

hist_plot_fordeling(valued_df['roof_ref']/valued_df['ghi_clear'], 'sekunddata')
hist_plot_fordeling(valued_df_resampled_1m['roof_ref']/valued_df_resampled_1m['ghi_clear'],'midlet minuttdata')
hist_plot_fordeling(valued_df_resampled_1h['roof_ref']/valued_df_resampled_1h['ghi_clear'], 'midlet timesdata')



# hist_plot_varighet(oie_lengths_by_day['oie_lengths'], 10)
# hist_plot_varighet(oie_lengths_by_day_1m['oie_lengths'], 5)
# hist_plot_varighet(oie_lengths_by_day_1h['oie_lengths'], 1)

# plt.hist(oie_lengths_by_day['oie_lengths'], bins=50, alpha=0.5, label='1s')
# plt.hist(oie_lengths_by_day_1m['oie_lengths'], bins=50, alpha=0.5, label='1min')
# plt.hist(oie_lengths_by_day_1h['oie_lengths'], bins=50, alpha=0.5, label='1h')
# plt.legend()
# plt.title('Distribusjon av overirradiansepisoder')
#
# plt.show()
#

# valued_df = merged_df[merged_df['ghi_extra'] > 1]

# test_df = valued_df[(valued_df.index.hour >= 6) &
#                     (valued_df.index.hour < 18)]
#
# fig, ax = plt.subplots()
# ghi, = ax.plot(test_df.index, test_df['ghi_clear'], label='GHI', color='orange')
# # toa, = ax.plot(test_df.index, test_df['ghi_extra'], label='TOA')
# roof_ref = ax.scatter(test_df.index, test_df['roof_ref'], s=1, label='Målte irradiansverdier')
# #df_avg_10s = test_df['sol_h'].resample('10s').mean().dropna()
# df_avg_60s = test_df['roof_ref'].resample('60s').mean().dropna()
# df_avg_1h = test_df['roof_ref'].resample('1h').mean().dropna()
# #sol_h_10s = ax.plot(df_avg_10s.index, df_avg_10s, color='green', label='10s')
# roof_ref_1m = ax.plot(df_avg_60s.index, df_avg_60s, color='red', label='1m')
# roof_ref_1h = ax.plot(df_avg_1h.index, df_avg_1h, color='blue', label='1h')
# # plt.xticks(test_df['time'][0::3600], rotation=70)
# ax.legend(loc='upper left')
# # plt.draw()
# plt.tight_layout()
# plt.show()

