import pvlib
import pandas as pd
from matplotlib import pyplot as plt

URL = 'api.soda-solardata.com'


# Ås: 59.660163, 10.782033, 93 moh
# Kjeller: 59.973238, 11.051619, 108 moh
# Finse: 60.593531, 7.524547, 1222 moh

latitude = 59.973238
longitude = 11.051619
start = '2021-08-01'
end = '2023-10-01'
email = 'simon.bakkejord@nmbu.no'
identifier = 'mcclear'
altitude = 108
time_step = '1min'
time_ref = 'UT'
verbose = False
integrated = False
label = 'left'
map_variables = True
server = pvlib.iotools.sodapro.URL
timeout = 30

raw_data = pvlib.iotools.get_cams(latitude, longitude, start, end, email, identifier, altitude, time_step,
                              time_ref, verbose, integrated, label, map_variables, server, timeout)

temp_path = 'C:\\Users\\simon\\OneDrive - Norwegian University of Life Sciences\\Documents\\Masteroppgave ' \
            'v24\\Programmering\\måledata\\temp\\mcclear_kjeller_2021_2023.csv '

raw_data[0].to_csv(path_or_buf=temp_path)
