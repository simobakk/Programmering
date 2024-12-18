import pvlib
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import time
import pickle


# Må skille på om data vil hentes fra tak eller fra solpark når data hentes - 'Roof' og 'Solpark'
data_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata'
mcclear_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\temp\\mcclear_kjeller_22-23.csv'


with open(data_path + '\\kjeller\\Solpark\\solpark_albedo_downwell_irr_20210820_20230925_reduced.pkl', 'rb') as pickle_in:
    data = pickle.load(pickle_in)

start = "2023-08-09 00:00:00"
end = "2023-08-10 00:00:00"
start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()

series = data.loc[start_date:end_date]

# print(len(series))
# series.reindex(pd.date_range(start=start, end=end, freq='S'))
# print(len(series))

# df = data.reset_index(name='sol_a_upwell')

raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=2)
raw_cs_df.set_index('TIMESTAMP', inplace=True)
old_df = raw_cs_df.loc[start_date:end_date]
cs_df = old_df.drop('Observation period', axis=1)

cs_df.astype(float)
cs_filled_df = cs_df.reindex(pd.date_range(start=start_date, end=end, freq='s'))
cs_filled_df.index = pd.to_datetime(cs_filled_df.index)
cs_filled_df.interpolate(method='polynomial', order=2, inplace=True)

full_df = pd.merge(series, cs_filled_df, left_index=True, right_index=True, how='outer')

# test_df = full_df[(full_df['index'] >= start) & (full_df['index'] <= end)]

fig, ax = plt.subplots()
measured = ax.scatter(full_df.index, full_df['solpark_albedo_downwell_irr'], s=1, label='IR_S', color='red')
ghi, = ax.plot(full_df.index, full_df['ghi_clear'], label='GHI')
toa, = ax.plot(full_df.index, full_df['ghi_extra'], label='TOA')
# dni, = ax.plot(full_df.index, full_df['dni_clear'], label='DNI')

ax.legend()
plt.show()