import pvlib
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import time
import pickle
from itertools import groupby


start = "2022-01-01 00:00:00"
end = "2022-11-30 23:59:00"
start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
end_date = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()

# Må skille på om data vil hentes fra tak eller fra solpark når data hentes - 'Roof' og 'Solpark'
data_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata'
mcclear_path = data_path + '\\temp\\mcclear_kjeller_2021-2023.csv'
outpath = data_path + '\\plots\\kjeller'

# Henter datasettet som er redusert til måneder med faktiske målinger.
with open(data_path + '\\kjeller\\Roof\\OEhor_irr_20210101_20231231.pkl', 'rb') as pickle_in:
    roof_ref_cell = pickle.load(pickle_in)

full_roof_irradiation_df = roof_ref_cell
full_roof_irradiation_df.rename(columns={'OEhor_irr': 'ghi_measured'}, inplace=True)
full_roof_irradiation_df.set_index('date_time', inplace=True)
full_roof_irradiation_df.sort_index(inplace=True)
roof_irradiation_df = full_roof_irradiation_df.loc[start_date:end_date]
# roof_downwell_reduced = roof_downwell.resample('10s').mean()

# Laster inn clearskymodell og fikser opp i strukturen
raw_cs_df = pd.read_csv(mcclear_path)
raw_cs_df.rename( columns={'Unnamed: 0':'TIMESTAMP'}, inplace=True )
raw_cs_df['TIMESTAMP'] = pd.to_datetime(raw_cs_df['TIMESTAMP'])
raw_cs_df['TIMESTAMP'] = raw_cs_df['TIMESTAMP'].dt.tz_localize(None)
raw_cs_df['TIMESTAMP'] += pd.Timedelta(hours=1)
raw_cs_df.set_index('TIMESTAMP', inplace=True)

# Korter ned CS-modell litt til
actual_df = raw_cs_df.loc[start:end]
cs_df = actual_df.drop('Observation period', axis=1)
# cs_df.astype(float)

# Interpolerer verdier i modellen, slik at hvert målepunkt har en tilhørende verdi
cs_interpolated_df = cs_df.reindex(pd.date_range(start=start_date, end=end, freq='s'))
cs_interpolated_df.index = pd.to_datetime(cs_interpolated_df.index)
cs_interpolated_df.interpolate(method='polynomial', order=2, inplace=True)
cs_interpolated_df['date'] = cs_interpolated_df.index.date
cs_interpolated_df['time'] = cs_interpolated_df.index.time.astype(str)
print(cs_interpolated_df.head())

# Lager en liste over unike tidspunkt i løpet av dagen
unique_time = [f"{hour:02d}:{minute:02d}:{second:02d}"
                       for hour in range(24)
                       for minute in range(60)
                       for second in range(60)]


# Slår sammen df, og velger ut kun dagstid måletidspunkt for å korte ned datasettet litt
merged_df = roof_irradiation_df.merge(cs_interpolated_df, left_index=True, right_index=True)
unique_days = merged_df['date'].unique()
# Denne under fjerner ubrukelig nan verdi som kommer først av en eller annen grunn.
unique_days = unique_days[1:]
test_days = unique_days[:5]
active_irradiation_df = merged_df[merged_df['ghi_clear'] > 0]


# def save_plots_kjeller(unique_days, unique_time, active_irradiation_df, outpath):
time_of_day_df = pd.DataFrame({'time': unique_time})

for date in test_days:
    day_data = active_irradiation_df[active_irradiation_df['date'] == date]
    daily_df = pd.merge(time_of_day_df, day_data, on='time', how='left')
    day_data.dropna(inplace=True)

    fig, ax = plt.subplots()
    ax.plot(daily_df['time'], daily_df['ghi_clear'], label='GHI Clear', color = 'orange')
    ax.scatter(day_data['time'], day_data['ghi_measured'], label='GHI Measured', color = 'blue')

    ax.set_xlabel('Tid')
    ax.set_ylabel('GHI')
    ax.set_title(f'Irradians for {date}')
    ax.legend()
    ax.set_xticks(unique_time[0::3600*3], labels = unique_time[0::3600*3], rotation = 20)
    fig.tight_layout()
    plt.show()

    #fig.savefig(os.path.join(outpath, f'new plot_roof_ref_{date}'.format(date)))
    #plt.close(fig)  # Close the figure to free memory

#save_plots_kjeller(unique_days, unique_time, active_irradiation_df, outpath)


active_irradiation_df['oie'] = active_irradiation_df['ghi_clear'] > active_irradiation_df['ghi_measured']

def get_max_measured_and_expected(group):
    if group['ghi_measured'].notna().any():
        max_idx = group['ghi_measured'].idxmax()
        max_value = group.loc[max_idx, 'ghi_measured']
        expected_value = group.loc[max_idx, 'ghi_clear']
        return pd.Series([max_value,expected_value])
    else:
        return pd.Series([np.nan, np.nan])

max_active_df = active_irradiation_df.groupby('date', group_keys=False).apply(get_max_measured_and_expected, include_groups=False).reset_index()
max_active_df.columns = ['date', 'max_value', 'expected_max_value']
max_active_df['max_ratio'] = max_active_df['max_value']/max_active_df['expected_max_value']

simple_stats_active_df = active_irradiation_df.groupby('date')['ghi_measured'].agg(['median','mean','sum']).reset_index()
count_full_df = merged_df.groupby('date')['ghi_measured'].count().reset_index()
count_full_df.columns = ['date', 'count']
stats_active_df = max_active_df.merge(simple_stats_active_df, on='date', how='left')
stats_active_df = stats_active_df.merge(count_full_df, on='date', how='left')
stats_active_df.columns = ['date', 'max_value', 'expected_max_value', 'max_ratio', 'median', 'mean', 'sum_measured', 'count']

stats_active_df['health'] = stats_active_df['count']/(24*60*60)
stats_model_df = active_irradiation_df.groupby('date')['ghi_clear'].sum().reset_index()
stats_model_df.columns = ['date', 'sum_model']

def calculate_oie_lengths (group):
    series = group['oie']
    oie_lengths = [sum(1 for _ in g) for k, g in groupby(series) if k]
    return pd.Series({'oie_lengths': oie_lengths})

oie_lengths_by_day = active_irradiation_df.groupby('date', group_keys=False).apply(calculate_oie_lengths, include_groups=False).reset_index()
oie_lengths_by_day.columns = ['date', 'oie_lengths']

oie_count = active_irradiation_df.groupby('date')['oie'].sum().reset_index()
oie_count.columns = ['date', 'oie_counts']

daily_stats_df = oie_count.merge(oie_lengths_by_day, on='date', how='left')
daily_stats_df = daily_stats_df.merge(stats_active_df, on='date', how='left')
daily_stats_df = daily_stats_df.merge(stats_model_df, on='date', how='left')
daily_stats_df['measured/expected'] = daily_stats_df['sum_measured']/daily_stats_df['sum_model']
daily_stats_df['oie_counts'] = daily_stats_df['oie_counts'].fillna(0).astype(int)

