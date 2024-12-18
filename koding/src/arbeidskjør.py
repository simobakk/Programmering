import pvlib
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import time

# IR_S: Infrared radiation data from the South (possibly indicating solar radiation intensity or temperature).
# Udc_N: Direct current (DC) voltage from the North.
# Idc_N: DC current from the North.
# Udc_S: DC voltage from the South.
# Idc_S: DC current from the South.
# Uac_S: Alternating current (AC) voltage from the South.
# Iac_S: AC current from the South.
# Pac_S: AC power from the South.
# VA_S: Apparent power from the South.
# VAr_S: Reactive power from the South.
# PF_S: Power factor from the South.
# Hz_S: Frequency from the South.
# Uac_N: AC voltage from the North.
# Iac_N: AC current from the North.
# Pac_N: AC power from the North.
# VA_N: Apparent power from the North.
# VAr_N: Reactive power from the North.
# PF_N: Power factor from the North.
# Hz_N: Frequency from the North.

# Paths
mcclear_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\temp\\mcclear_søråsjordet_2023.csv'
outpath = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\plots\\søråsjordet'

raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
#raw_cs_df['TIMESTAMP_copy'] = raw_cs_df['TIMESTAMP']
#raw_cs_df['TIMESTAMP_copy'] = raw_cs_df['TIMESTAMP_copy'].dt.tz_convert('Europe/Berlin')
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=1)
raw_cs_df.set_index('TIMESTAMP', inplace=True)

# Henter innstrålingsdata, her fra Søråsjordet
current_file_path = 'C:/Users/simon/OneDrive - Norwegian University of Life Sciences/Documents/Masteroppgave v24/Programmering'
meas_file_path = current_file_path + '/måledata/søråsjordet/CR1000_PVfast_04.12.23.dat'

sj_meas_df = pd.read_csv(meas_file_path, skiprows=1, low_memory=False)

# Behandler datasettet
sj_meas_df = sj_meas_df.iloc[2:]
sj_meas_df.iloc[:, 1] = sj_meas_df.iloc[:, 1].astype(int)
sj_meas_df.iloc[:, 2:21] = sj_meas_df.iloc[:, 2:21].astype(float)
sj_meas_df['TIMESTAMP'] = pd.to_datetime(sj_meas_df['TIMESTAMP'])
sj_meas_df['date'] = sj_meas_df['TIMESTAMP'].dt.date
sj_meas_df['time'] = sj_meas_df['TIMESTAMP'].dt.time.astype(str)
sj_meas_df['time'] = sj_meas_df['time'][:-3]
sj_meas_df['index'] = sj_meas_df['TIMESTAMP']
sj_meas_df.set_index('index', inplace=True)

measure_start = pd.to_datetime(sj_meas_df.index[0].strftime('%Y-%m-%d') + ' 00:00:00')
measure_end = pd.to_datetime(sj_meas_df.index[-1].strftime('%Y-%m-%d') + ' 23:59:59')

# Finner clear sky model for plots, og behandler datasettet
sliced_raw_cs_df = raw_cs_df.loc[measure_start:measure_end]
cs_df = sliced_raw_cs_df[['ghi_extra', 'ghi_clear', 'dni_clear']]


# Slår sammen de to datasettene, og får en dataframe med all data sidestilt
merged_df = pd.concat([sj_meas_df[['IR_S', 'TIMESTAMP']], cs_df[['ghi_clear', 'ghi_extra', 'dni_clear']]],
                      axis=1) #, keys=['sj_11223_df', 'cs_df'])

# Lager lister for å iterere over dager og tidspunkter, og se hvilke dager som skal ses på
unique_days = [measure_start + timedelta(days=i) for i in range((measure_end - measure_start).days + 1)]
unique_time = [f"{hour:02d}:{minute:02d}:{second:02d}"
               for hour in range(24)
               for minute in range(60)
               for second in range(60)]

testdate = unique_days[0].date()
test_df = merged_df[merged_df.index.date == testdate]
unique_timestamps = [datetime.combine(testdate, datetime.strptime(time_str, '%H:%M:%S').time())
                     for time_str in unique_time]
full_df = pd.DataFrame(index=unique_timestamps)

merged_test_df = pd.concat([test_df, full_df], axis=1)
merged_test_df['TIME'] = unique_time

fig, ax = plt.subplots()
measured = ax.scatter(merged_test_df['TIME'], merged_test_df['IR_S'], s=1, label='IR_S', color='red')
ghi, = ax.plot(merged_test_df['TIME'][0::60], merged_test_df['ghi_clear'][0::60], label='GHI')
toa, = ax.plot(merged_test_df['TIME'][0::60], merged_test_df['ghi_extra'][0::60], label='TOA')
dni, = ax.plot(merged_test_df['TIME'][0::60], merged_test_df['dni_clear'][0::60], label='DNI')

max_ir = merged_test_df['IR_S'].max()
max_cs_extra = merged_test_df['ghi_extra'].max()
buffer = 0.1
ax.set_ylim(0, max(100, max(max_cs_extra, max_ir)) * (1 + buffer))

# Set every second (2h * 60 min * 60 "1 sec"-ticks) hour as ticks
plt.xticks(merged_test_df['TIME'][0::7200], rotation=70)

ax.set_xlabel('Time (h)')
ax.set_ylabel('Irradiance (W/m^2)')
ax.set_title(f'Clear Sky models and measured data for {testdate}')

ax.legend()
plt.draw()
plt.tight_layout()
plt.show()

max_cs_clear = merged_test_df['ghi_clear'].max()
max_clear_idx = merged_test_df['ghi_clear'].idxmax()
max_extra_idx = merged_test_df['ghi_extra'].idxmax()
# print(f'Plotter nå for datafilen {filename}')
# for day in range(len(unique_days)):
#     testdate = unique_days[day].date()
#     test_df = merged_df[merged_df.index.date == testdate]
#     unique_timestamps = [datetime.combine(testdate, datetime.strptime(time_str, '%H:%M:%S').time())
#                          for time_str in unique_time]
#
#     start_of_day = unique_timestamps[0].strftime('%Y-%m-%d %H:%M:%S')
#     end_of_day = unique_timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')
#     full_range = pd.date_range(start=start_of_day, end=end_of_day, freq='S')
#     full_df = pd.DataFrame(index=unique_timestamps)
#
#     merged_test_df = pd.concat([test_df, full_df], axis=1)
#     merged_test_df['TIME'] = unique_time
#     # merged_test_df.reset_index(inplace=True)
#
#     ghi.set_data(merged_test_df['TIME'][0::60], merged_test_df['ghi_clear'][0::60])
#     toa.set_data(merged_test_df['TIME'][0::60], merged_test_df['ghi_extra'][0::60])
#     measured.set_offsets(merged_test_df[['TIME', 'IR_S']])
#
#     max_ir = merged_test_df['IR_S'].max()
#     max_cs = merged_test_df['ghi_extra'].max()
#     buffer = 0.1
#     ax.set_ylim(0, max(100, max(max_cs, max_ir)) * (1 + buffer))
#     ax.set_title(f'Clear Sky models and measured data for {testdate}')
#
#     plt.draw()
#     plt.tight_layout()
#     fig.savefig(os.path.join(outpath, 'sorasjordet_{0}_{1}'.format(testdate, day)))
