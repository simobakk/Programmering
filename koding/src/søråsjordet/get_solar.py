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
programmering_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\'
mcclear_path = programmering_path + 'måledata/temp/mcclear_søråsjordet_2023.csv'
outpath = programmering_path + 'måledata/analyser/søråsjordet'

# Finner solvinkler i tidsrommet
# Ås: 59.660163, 10.782033
alt = 93
latitude = 59.660163
longitude = 10.782033

raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])

solar_positions = pvlib.solarposition.get_solarposition(raw_cs_df['TIMESTAMP'], latitude, longitude, altitude=alt)
raw_cs_df = pd.concat([raw_cs_df, solar_positions], axis=1)

raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=1)
raw_cs_df['date'] = raw_cs_df['TIMESTAMP'].dt.date
raw_cs_df.set_index('TIMESTAMP', inplace=True)

# Henter innstrålingsdata, her fra Søråsjordet
dir_file_path = programmering_path + '/måledata/søråsjordet'

# Klargjør dataframe
columns = ["TIMESTAMP","RECORD","IR_S","Udc_N","Idc_N","Udc_S","Idc_S","Uac_S","Iac_S","Pac_S",
           "VA_S","VAr_S","PF_S","Hz_S","Uac_N","Iac_N","Pac_N","VA_N","VAr_N","PF_N","Hz_N"]
combined_df = pd.DataFrame(columns=columns)

for filename in os.listdir(dir_file_path):
    if filename.endswith('.dat'):
        meas_file_path = os.path.join(dir_file_path, filename)
        df = pd.read_csv(meas_file_path, skiprows=4, names = columns, low_memory=False)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    break

# Setter første kolonner (med timestamps) som int for å få ut tid og dato.
combined_df.iloc[:, 1] = combined_df.iloc[:, 1].astype(int)
combined_df.iloc[:, 2:21] = combined_df.iloc[:, 2:21].astype(float)

combined_df['TIMESTAMP'] = pd.to_datetime(combined_df['TIMESTAMP'])
combined_df['date'] = combined_df['TIMESTAMP'].dt.date
combined_df['time'] = combined_df['TIMESTAMP'].dt.time.astype(str)
combined_df['time'] = combined_df['time'][:-3]
combined_df['timestamp'] = combined_df['TIMESTAMP']
combined_df.set_index('TIMESTAMP', inplace=True)

# Kun cs data fra dager som har måledata
filtered_cs_df = raw_cs_df[raw_cs_df['date'].isin(combined_df['date'].unique())]

merged_df = combined_df.merge(filtered_cs_df, how = 'outer', left_index=True, right_index=True)
#merged_df.set_index('TIMESTAMP', inplace=True)

merged_df['timezoned'] = merged_df.index
merged_df['timezoned'].tz_localize(tz='UTC')
merged_df['timezoned'] = merged_df['timezoned'].dt.tz_localize('UTC').dt.tz_convert('Europe/Oslo')
#merged_df.set_index(merged_df['timezoned'], inplace=True)

print(merged_df['timezoned'].head())

# Lager lister for å iterere over dager og tidspunkter, og se hvilke dager som skal ses på
unique_days = merged_df['date_y'].unique()
unique_time = [f"{hour:02d}:{minute:02d}:{second:02d}"
               for hour in range(24)
               for minute in range(60)
               for second in range(60)]

fig1, ax1 = plt.subplots()
measured = ax1.scatter(merged_df.index, merged_df['IR_S'], s=1, label='IR_S', color='red')
ghi = ax1.scatter(merged_df.index, merged_df['ghi_clear'], s=1, color='blue')
ax2 = ax1.twinx()
zenith, = ax2.plot(merged_df.index, merged_df['apparent_zenith'], label='ap_zenith', color='blue')
elevation = ax2.plot(merged_df.index, merged_df['apparent_elevation'], label='ap_elevation', color='green')
plt.show()

