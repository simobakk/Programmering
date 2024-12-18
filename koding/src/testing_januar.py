import pvlib
import os
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

mcclear_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\temp\\raw_data.csv '
save_fig_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\plots'

full_cs_df = pd.read_csv(mcclear_path)


cs_df_two = full_cs_df[['Observation period', 'ghi_extra', 'ghi_clear']]

current_file_path = 'C:/Users/simon/OneDrive - Norwegian University of Life Sciences/Documents/Masteroppgave v24/Programmering'
finse_22_file_path = current_file_path + '/måledata/Finse/Finseflux_level1_10sec_2022.csv'
f_22_df = pd.read_csv(finse_22_file_path)

start = '2022-01-02 00:00:00'
end = '2022-01-02 23:59:50'



f_22_jan1_jan2_df = f_22_df[(f_22_df['TIMESTAMP'] >= start)&(f_22_df['TIMESTAMP'] <= end)]

cs_df_two.info()
f_22_jan1_jan2_df.info()

cs_df_two.loc[:, 'Observation period'] = pd.to_datetime(cs_df_two['Observation period'].astype(str).str.split('/').str[0])

midpoint = len(cs_df_two) // 2

cs_df = cs_df_two.loc[midpoint:]
f_22_jan1_jan2_df.loc[:, 'TIMESTAMP'] = pd.to_datetime(f_22_jan1_jan2_df['TIMESTAMP'])

# cs_df.set_index(cs_df.iloc[:, 0], inplace=True)
# f_22_jan1_jan2_df.set_index('TIMESTAMP', inplace=True)

cs_df.info()
# f_22_jan1_jan2_df.info()

plt.plot(cs_df['Observation period'], cs_df['ghi_clear'], label='GHI')
plt.plot(cs_df['Observation period'], cs_df['ghi_extra'], label='TOA')
plt.scatter(f_22_jan1_jan2_df['TIMESTAMP'], f_22_jan1_jan2_df['SWin'], s=1, label='SWin', color='red')
plt.title('Clear Sky models and actual data for Jan 1st and 2nd 2022')
plt.xlabel('Date (or black line)')
plt.ylabel('Irradiance W/m^2')
plt.xticks(cs_df['Observation period'], rotation=70)
plt.legend()
plt.tight_layout()
plt.show()
