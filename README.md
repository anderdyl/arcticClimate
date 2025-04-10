# Arctic-TESLA

ArcticTESLA is a collection of Python3 and Matlab functions for generating stochastic wave and water level scenarios at coastal locations in Northern Alaska on the Chukchi and Beaufort Seas.
The package creates new time series of forcing conditions by generating new possible synoptic weather chronologies. 
The workflow identifies historical synoptic weather patterns, and the meteorologic and oceanic conditions that occurred during those weather systems.
Markov chains for each pattern, as well as their likelihood of occurrence conditional dependent on large scale climate indices, are then used in monte carlo simulations.
All codes are used in Anderson and Cohn (in review), with some functions adopted from https://github.com/teslakit/teslakit or from Anderson et al. (2019) and the references within for more details.

# Data
### Sea Level Pressure Fields
Python3 and Matlab functions are provided for loaded CFSR SLP fields that are downloaded to a local directory. The user must download all monthly files since 1979 found at the following links:

https://rda.ucar.edu/datasets/ds093.1/#description
https://rda.ucar.edu/datasets/ds094.1/#description

You will need to chose 'All available' under the data tab, at full resolution in lat/lon, and click the checkbox to convert the download from grib to netcdf inorder to work with the built-in functions in this library.

### Sea Ice Concentration Fields
The user will need to download all northern hemipshere .bin SIC files from the National Snow and Ice Data Center at https://nsidc.org/data/nsidc-0081/versions/2. 
The nc2bin_siconc.py function within dataDownloads will then convert all files to the format expected by sic.py.

### Local Waves, Winds, and Temperatures
To use ERA5 hindcasts, this package requires access to the online Thredds server hosted by Copernicus.

1. Create an account with Copernicus by signing up here.
2. Once you have an account, sign in to your Copercius account here and note the UID and API key at the bottom of the page.
3. Paste the code snippet below into your terminal, replacing 'UID' and 'API' with those from step 2:

(echo 'url: https://cds.climate.copernicus.eu/api/v2';
  echo 'key: UID:API';
  echo 'verify: 0';
   ) >> ~/.cdsapirc

The above command creates the file ~/.cdsapirc with your API key, which is necessary to use the CDS API. As a sanity check, use more ~/.cdsapirc to ensure everything appears correct.

### Water Levels
Use export_local_tides_alaska.m to download the latest NOAA tide gauge data from around Alaska and interpolate it to Point Hope, AK.

### Climate Variables: SST and OLR
Large-scale climate is accounted for by spatial patterns of the nearby ocean's sea surface temperature (SST) pattern at an annual scale. The awt.py file expects data in the yearly format available at https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/index.html?Set-Language=en. 
And the mjo.py script expects data in the format available at: https://iridl.ldeo.columbia.edu/SOURCES/.BoM/.MJO/.RMM/index.html?Set-Language=en.

# Methods

1. After downloading CFSR SLPs and extracting with CFSR_extractSLPs_rectify_cropLand.m, run dwts.py.
2. After downloading NSIDC SICs and converting with nc2bin_siconc.py, run sic.py.
3. After downloading ERA5 waves and NOAA tides, run hydrographs.py
4. After downloading ERSSTv5 SSTs, run awt.py
5. After downloading BOM MJO indices, run mjo.py
6. Run copulas.py
7. Run futureIceSimulations.py
8. Run futureSLPsimulations.py


Anderson, D. and N. Cohn (in review) Future coastal tundra loss due to compounding environmental changes in Alaska.

Anderson, D., A. Rueda, L. Cagigal, J. Antolinez, F. Mendez, and P. Ruggiero. (2019) Time-varying Emulator for Short and Long-Term Analysis of Coastal Flood Hazard Potential. Journal of Geophysical Research: Oceans, 124(12), 9209-9234. https://doi.org/10.1029/2019JC015312