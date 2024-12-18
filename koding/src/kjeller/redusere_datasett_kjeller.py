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

with open(data_path + '\\kjeller\\Roof\\roof_albedo_downwell_irr_20210820_20230925.pkl', 'rb') as pickle_in:
    data = pickle.load(pickle_in)

# Ser hvor det er data fra, lagrer kun relevant tidsrom
start = "2022-08-01 00:00:00"
end = "2023-09-26 23:59:59"

data_reduced = data.loc[start:]

filepath = data_path + "\\kjeller\\Roof\\roof_albedo_downwell_irr_20210820_20230925_reduced.pkl"
data_reduced.to_pickle(filepath)


