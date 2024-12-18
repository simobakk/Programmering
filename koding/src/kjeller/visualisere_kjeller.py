import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import time
from scipy.stats import linregress

# Timer
start_time = time.time()

# Må skille på om data vil hentes fra tak eller fra solpark når data hentes - 'Roof' og 'Solpark'
data_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata'
mcclear_path = data_path + '\\temp\\mcclear_kjeller_2021_2023.csv'
outpath = data_path + '\\plots\\kjeller'

# Setter tidspunkt for datasettet
start = "2022-08-01 00:00:00"
end = "2023-09-30 23:59:59"

start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# with open(data_path + '\\kjeller\\Solpark\\solpark_albedo_downwell_irr_20210820_20230925_reduced.pkl', 'rb') as pickle_in:
#     solar_adown_data_raw = pickle.load(pickle_in)
# solar_adown_data = solar_adown_data_raw[start:end].to_frame('sol_a')

with open(data_path + '\\kjeller\\Solpark\\KZPR_IRHoSunP_20210820_20230925.pkl', 'rb') as pickle_in:
    solar_sunh_data_raw = pickle.load(pickle_in)
solar_sunh_data = solar_sunh_data_raw[start:end].to_frame('sol_h')

# with open(data_path + '\\kjeller\\Roof\\roof_albedo_downwell_irr_20210820_20230925_reduced.pkl', 'rb') as pickle_in:
#     roof_adown_data_raw = pickle.load(pickle_in)
# roof_adown_data = roof_adown_data_raw[start:end].to_frame('roof_ad')
#
with open(data_path + '\\kjeller\\Roof\\OEhor_irr_20210101_20231231.pkl', 'rb') as pickle_in:
    roof_ref_data_raw = pickle.load(pickle_in)
roof_ref_data_raw.set_index('date_time', inplace=True)
roof_ref_data_series = roof_ref_data_raw['OEhor_irr']
roof_ref_data = roof_ref_data_series[start:end].to_frame('roof_ref')
#
# with open(data_path + '\\kjeller\\Roof\\roof_albedo_upwell_irr_20210101_20231005.pkl', 'rb') as pickle_in:
#     roof_au_data_raw = pickle.load(pickle_in)
# roof_au_data_raw.set_index('date_time', inplace=True)
# roof_au_data_series = roof_au_data_raw['roof_albedo_upwell_irr']
# roof_au_data = roof_au_data_series[start:end].to_frame('roof_au')
#
# with open(data_path + '\\kjeller\\Roof\\solys_dhi_irr_20210101_20231231.pkl', 'rb') as pickle_in:
#     roof_dhi_data_raw = pickle.load(pickle_in)
# roof_dhi_data_raw.set_index('date_time', inplace=True)
# roof_dhi_data_series = roof_dhi_data_raw['solys_dhi_irr']
# roof_dhi_data = roof_dhi_data_series[start:end].to_frame('roof_dhi')
#
# with open(data_path + '\\kjeller\\Roof\\solys_dni_irr_20210101_20231231.pkl', 'rb') as pickle_in:
#     roof_dni_data_raw = pickle.load(pickle_in)
# roof_dni_data_raw.set_index('date_time', inplace=True)
# roof_dni_data_series = roof_dni_data_raw['solys_dni_irr']
# roof_dni_data = roof_dni_data_series[start:end].to_frame('roof_dni')

# Laster inn cs-modellen
raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=2)
raw_cs_df.set_index('TIMESTAMP', inplace=True)

columns_to_drop = ['Observation period', 'bhi_clear', 'dni_clear', 'dhi_clear']
cs_df = raw_cs_df.drop(columns_to_drop, axis=1).loc[start_date:end_date]

# Fyller inn cs-modellen
cs_df.astype(float)
cs_filled_df = cs_df.reindex(pd.date_range(start=start_date, end=end, freq='s'))
cs_filled_df.index = pd.to_datetime(cs_filled_df.index)
cs_filled_df.interpolate(method='polynomial', order=2, inplace=True)
# cs_filled_df['date'] = cs_filled_df.index.date
# cs_filled_df['time_cs'] = cs_filled_df.index.time.astype(str)

#merged_df = pd.merge(cs_filled_df, solar_adown_data, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(cs_filled_df, solar_sunh_data, left_index=True, right_index=True, how='outer')
# merged_df = pd.merge(merged_df, roof_adown_data, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, roof_ref_data, left_index=True, right_index=True, how='outer')
# merged_df = pd.merge(merged_df, roof_au_data, left_index=True, right_index=True, how='outer')
# merged_df = pd.merge(merged_df, roof_dhi_data, left_index=True, right_index=True, how='outer')
# merged_df = pd.merge(merged_df, roof_dni_data, left_index=True, right_index=True, how='outer')

# merged_df['sol_a_10s_smoothed'] = merged_df['sol_a'].rolling(window=10, center=True).mean()
# Removed below due to date_list bein the same
# merged_df_unique_dates = merged_df['date'].unique()

unique_time = [f"{hour:02d}:{minute:02d}:{second:02d}"
                       for hour in range(24)
                       for minute in range(60)
                       for second in range(60)]

test_df = merged_df[merged_df.index.date == date_list[0]]
test_df.loc[:, 'time'] = unique_time
# start_index = test_df.index.searchsorted(10800)
# end_index = test_df.index.searchsorted(10800, side='right') - 1
# test__df = test_df.iloc[start_index:end_index+1]

# test_df = test_df[(test_df['time'].dt.hour >= 3) &
#                   (test_df['time'].dt.hour < 21)]

print('before plots')

# fig, ax = plt.subplots()
# ghi, = ax.plot(test_df['time'], test_df['ghi_clear'], label='GHI')
# toa, = ax.plot(test_df['time'], test_df['ghi_extra'], label='TOA')
# # dni, = ax.plot(test_df['time'], test_df['dni_clear'], label='DNI')
# # dhi, = ax.plot(test_df['time'], test_df['dhi_clear'], label='DHI')
#
# print('cs done')
#
# # sol_a = ax.scatter(test_df['time'], test_df['roof_ref'], s=1, label='sol_a')
# sol_h = ax.scatter(test_df['time'], test_df['sol_h'], s=1, label='Irradians solpark', color = 'green')
# # roof_ad = ax.scatter(test_df['time'], test_df['roof_ad'], s=1, label='roof_ad')
# roof_ref = ax.scatter(test_df['time'], test_df['roof_ref'], s=1, label='Irradians takstasjon', color = 'red')
# # roof_au = ax.scatter(test_df['time'], test_df['roof_au'], s=1, label='roof_au')
# # roof_dhi = ax.scatter(test_df['time'], test_df['roof_dhi'], s=1, label='roof_dhi')
# # roof_dni = ax.scatter(test_df['time'], test_df['roof_dni'], s=1, label='roof_dni')
# print('measures done')
# plt.xticks(unique_time[0::3600*2], rotation=70)
#
# ax.legend(loc='upper left')
# plt.xlabel('Tid (hh:mm:ss)')
# plt.ylabel('Irradians (Wm^-2)')
#
# # plt.draw()
# plt.tight_layout()
# # plt.show()
# print('before loop')
#
# for date in date_list:
#     daily_df = merged_df[merged_df.index.date == date]
#     # daily_df['time'] = daily_df.index.time.astype(str)
#     daily_df = daily_df[~daily_df.index.duplicated(keep='first')]
#     daily_df.loc[:, 'time'] = unique_time
#
#     daily_df = daily_df[daily_df['ghi_extra'] > 1]
#
#     ghi.set_data(daily_df['time'], daily_df['ghi_clear'])
#     toa.set_data(daily_df['time'], daily_df['ghi_extra'])
#     # dni.set_data(daily_df.index, daily_df['dni_clear'])
#     # dhi.set_data(daily_df.index, daily_df['dhi_clear'])
#
#     # sol_a.set_offsets(daily_df[['time', 'roof_ref']])
#     sol_h.set_offsets(daily_df[['time', 'sol_h']])
#     # roof_ad.set_offsets(daily_df[['time', 'roof_ad']])
#     roof_ref.set_offsets(daily_df[['time', 'roof_ref']])
#     # roof_au.set_offsets(daily_df[['time', 'roof_au']])
#     # roof_dhi.set_offsets(daily_df[['time', 'roof_dhi']])
#     # roof_dni.set_offsets(daily_df[['time', 'roof_dni']])
#
#     print(daily_df['roof_ref'].max())
#     max_value = daily_df[['ghi_clear', 'ghi_extra', 'sol_h', 'roof_ref']].max().max()
#     ax.set_ylim(0, max(100, max_value) * 1.1)
#     #ax.set_ylim(0,1200)
#     plt.title(f"Irradians på Kjeller {date}")
#     plt.draw()
#     plt.tight_layout()
#     fig.savefig(os.path.join(outpath, 'roof_and_solar_{0}'.format(date)))
#     print(f"Saved date: {date}")


selected_df = merged_df[['sol_h', 'roof_ref']]
filtered_df = selected_df.dropna()

df_clean = filtered_df[(filtered_df['sol_h'] >= 0) & (filtered_df['roof_ref'] >= 0)]
dataset1_filtered = df_clean['sol_h'].values
dataset2_filtered = df_clean['roof_ref'].values
dataset3_filtered = df_clean['roof_ref'].values*1.1

slope, intercept, r_value, p_value, std_err = linregress(dataset1_filtered, dataset2_filtered)
dataset_2_corrected = slope * dataset2_filtered + intercept

print(f"The slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}")


r = np.corrcoef(dataset1_filtered, dataset2_filtered)[0, 1]
print(f"Pearson correlation coefficient: {r}")
mape = np.mean(np.abs((dataset1_filtered - dataset2_filtered) / dataset1_filtered)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
rmse = np.sqrt(np.mean((dataset1_filtered - dataset2_filtered)**2))
print(f"Root Mean Square Error (RMSE): {rmse}")


end_time = time.time()
exectution_time = end_time-start_time
print(f"Exection time: {exectution_time} seconds.")