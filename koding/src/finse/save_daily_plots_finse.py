import pvlib
import os
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
import time

# Timer
start_time = time.time()

# Lager paths først
current_file_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\'
mcclear_path = current_file_path + '/måledata/temp/mcclear_finse_2023.csv'
outpath = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\plots'

# Finner clear sky model for plots
raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
cs_df = raw_cs_df[['TIMESTAMP', 'ghi_extra', 'ghi_clear']]

# Henter innstrålingsdata, her fra Finse
finse_23_file_path = current_file_path + '/måledata/Finse/Finseflux_level1_10sec_2023.csv'
f_23_df = pd.read_csv(finse_23_file_path)

# Angir startpunkt og sluttpunkt for data som ønskes behandlet
start = '2023-05-01 00:00:00'
end = '2023-05-10 23:59:50'
f_23_limited_df = f_23_df[(f_23_df['TIMESTAMP'] >= start) & (f_23_df['TIMESTAMP'] <= end)]

# Lager en liste med date-objekter over hvilke dager å se på
f_23_limited_df['TIMESTAMP'] = pd.to_datetime(f_23_limited_df['TIMESTAMP'])
f_23_limited_df['date'] = f_23_limited_df['TIMESTAMP'].dt.date
f_23_limited_df['time'] = f_23_limited_df['TIMESTAMP'].dt.time.astype(str)

cs_df['TIMESTAMP'] = pd.to_datetime(cs_df['TIMESTAMP'])
cs_df['TIMESTAMP'] = cs_df['TIMESTAMP'].dt.tz_localize(None)
cs_df['date'] = cs_df['TIMESTAMP'].dt.date
cs_df['time'] = cs_df['TIMESTAMP'].dt.time.astype(str)
unique_days = f_23_limited_df['date'].unique().tolist()
unique_time = f_23_limited_df['time'].unique().tolist()

testdate = f_23_limited_df['date'].iloc[0]


test_cs = cs_df[cs_df['TIMESTAMP'].dt.date == testdate]
test_measured = f_23_limited_df[f_23_limited_df['TIMESTAMP'].dt.date == testdate]

fig, ax = plt.subplots()
measured = ax.scatter(unique_time, test_measured['SWin'], s=1, label='SWin', color='red')
ghi, = ax.plot(unique_time[0::6], test_cs['ghi_clear'], label='GHI', color = 'blue')
toa, = ax.plot(unique_time[0::6], test_cs['ghi_extra'], label='TOA', color = 'orange')

ax.set_xlabel('Tidspunkt (hh:mm:ss)')
ax.set_ylabel('Irradians (W/m^2)')
ax.set_title('"Clear Sky" models and measured data for testdate[2023-01-01]')
# Set every second (2h * 60 min * 6 "10 sec"-ticks) hour as ticks
plt.xticks(unique_time[0::720], rotation=70)

ax.legend()

plt.draw()
plt.tight_layout()
plt.show()

# plt.savefig(os.path.join(outpath, "finse_all.png"))

print(f'Lagrer plot til {len(unique_days)} dager')

for day in unique_days:
    filtered_cs = cs_df[cs_df['TIMESTAMP'].dt.date == day]
    filtered_measured = f_23_limited_df[f_23_limited_df['TIMESTAMP'].dt.date == day]
    ghi.set_data(filtered_cs['time'], filtered_cs['ghi_clear'])
    toa.set_data(filtered_cs['time'], filtered_cs['ghi_extra'])
    measured.set_offsets(filtered_measured[['time', 'SWin']])
    ax.set_title(f'"Clear Sky"-modeller og måledata på Finse for {day}')
    max_meas = filtered_measured['SWin'].max()
    max_cs = filtered_cs['ghi_extra'].max()
    buffer = 0.1

    ax.set_ylim(0, 1400)

    plt.draw()
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'aaa_finse_eksempel_{0}'.format(day)))

# Timer
end_time = time.time()
exectution_time = end_time-start_time
print(f"Exection time: {exectution_time} seconds.")

