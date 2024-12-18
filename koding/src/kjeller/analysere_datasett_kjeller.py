import pandas as pd
import numpy as np
import pickle

# Må skille på om data vil hentes fra tak eller fra solpark når data hentes - 'Roof' og 'Solpark'
data_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata'

# Setter tidspunkt for datasettet
start = "2021-01-01 00:00:00"
end = "2023-12-31 23:59:59"


# with open(data_path + '\\kjeller\\Solpark\\solpark_albedo_downwell_irr_20210820_20230925_reduced.pkl', 'rb') as pickle_in:
#     solar_adown_data_raw = pickle.load(pickle_in)
# solar_adown_data = solar_adown_data_raw[start:end].to_frame('sol_a')

# with open(data_path + '\\kjeller\\Solpark\\KZPR_IRHoSunP_20210820_20230925.pkl', 'rb') as pickle_in:
#     solar_sunh_data_raw = pickle.load(pickle_in)
# solar_sunh_data = solar_sunh_data_raw[start:end].to_frame('sol_h')

# with open(data_path + '\\kjeller\\Roof\\roof_albedo_downwell_irr_20210820_20230925_reduced.pkl', 'rb') as pickle_in:
#     roof_adown_data_raw = pickle.load(pickle_in)
# roof_adown_data = roof_adown_data_raw[start:end].to_frame('roof_ad')

with open(data_path + '\\kjeller\\Roof\\OEhor_irr_20210101_20231231.pkl', 'rb') as pickle_in:
    roof_ref_data_raw = pickle.load(pickle_in)
roof_ref_data_raw.set_index('date_time', inplace=True)
roof_ref_data_series = roof_ref_data_raw['OEhor_irr']
roof_ref_data = roof_ref_data_series[start:end].to_frame('roof_ref')

# with open(data_path + '\\kjeller\\Roof\\roof_albedo_upwell_irr_20210101_20231005.pkl', 'rb') as pickle_in:
#     roof_au_data_raw = pickle.load(pickle_in)
# roof_au_data_raw.set_index('date_time', inplace=True)
# roof_au_data_series = roof_au_data_raw['roof_albedo_upwell_irr']
# roof_au_data = roof_au_data_series[start:end].to_frame('roof_au')

# with open(data_path + '\\kjeller\\Roof\\solys_dhi_irr_20210101_20231231.pkl', 'rb') as pickle_in:
#     roof_dhi_data_raw = pickle.load(pickle_in)
# roof_dhi_data_raw.set_index('date_time', inplace=True)
# roof_dhi_data_series = roof_dhi_data_raw['solys_dhi_irr']
# roof_dhi_data = roof_dhi_data_series[start:end].to_frame('roof_dhi')

# with open(data_path + '\\kjeller\\Roof\\solys_dni_irr_20210101_20231231.pkl', 'rb') as pickle_in:
#     roof_dni_data_raw = pickle.load(pickle_in)
# roof_dni_data_raw.set_index('date_time', inplace=True)
# roof_dni_data_series = roof_dni_data_raw['solys_dni_irr']
# roof_dni_data = roof_dni_data_series[start:end].to_frame('roof_dni')


def calculate_nan_summary(dataframe, column_name):
    dataframe['Year'] = dataframe.index.year
    dataframe['Month'] = dataframe.index.month

    nan_summary = (
        dataframe.groupby(['Year', 'Month'])
        .agg(
            Total_Rows=(column_name, 'size'),
            NaN_Count=(column_name, lambda x: x.isna().sum())
        )
        .assign(NaN_Percentage=lambda x: (x['NaN_Count'] / x['Total_Rows']) * 100)
        .reset_index()
    )
    return nan_summary

nan_summary_roof_ref = calculate_nan_summary(roof_ref_data, 'roof_ref')

#
# print(raw_data.iloc[0])
# print(raw_data.iloc[1])
# print(raw_data.iloc[-1])
#
# # Bruk dersom datasett er et df, ignorer om settet er Series
# raw_data.set_index('date_time', inplace=True)
# data = raw_data['solys_dhi_irr']
# # data = raw_data
#
# data_validate = []
# n_entries = []
# nan_list = []
# nan_count = -1
# prev_idx = 0
#
# for idx in range(len(data)):
#     datestrip = data.index[idx].strftime("%Y-%m-%d")
#     if np.isnan(data.iloc[idx]):
#         nan_count += 1
#     if datestrip not in data_validate:
#         data_validate.append(datestrip)
#         if not n_entries:
#             n_entries.append(idx)
#         else:
#             n_entries.append(idx-prev_idx)
#             nan_list.append(nan_count)
#             nan_count = 0
#         prev_idx = idx
#         print(datestrip)
#
#
# n_entries.append(len(data)-sum(n_entries))
# nan_list.append(nan_count+1)
# n_entries.pop(0)
#
# data_frame = {
#     "entries": n_entries,
#     "nan_values": nan_list
# }
# df = pd.DataFrame(data_frame, index = data_validate)
#
# filepath = data_path + "\\analyser\\kjeller\\solys_dhi_irr_20210101_20231231_analyzed.csv"
# df.to_csv(filepath)
# #
# # # Ser hvor det er data fra, lagrer kun relevant tidsrom
# # #start = "2022-10-07 00:00:00"
# #
# # #end = "2023-09-26 23:59:59"
# #
