import irradpy
import multiprocessing
import numpy as np

# Important Note: If you're using windows, make sure to wrap the function.
if __name__ == "__main__":
    multiprocessing.freeze_support()
    irradpy.downloader.run(auth={"uid":"USERNAME", "password": "PASSWORD"})

# 60.59353376582669, 7.524296919342721
# Download All Data From 2018-01-01 To 2018-01-02
irradpy.downloader.run(auth={"uid":"simon.bakkejord@nmbu.no", "password": "eaEcenturion27!"},
    initial_year=2022, final_year=2022,
    initial_month=1, final_month=12,
    initial_day=1, final_day=31,
    lat_1=60, lat_2=61,
    lon_1=7, lon_2=8,
    verbose=True,
    thread_num=20, connection_num=2
    )

np.array

time_delta = 10  # minute
timedef = [('2022-01-01T00:00:00', '2022-12-31T00:00:00')]
time = irradpy.model.timeseries_builder(timedef, time_delta, np.size(latitudes))
irradpy.model.ClearSkyREST2v5(latitudes, longitudes, elevations, time, dataset_dir).REST2v5()