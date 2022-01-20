import pandas as pd
import numpy as np
import re
import os
import datetime as dt
import zipfile
import shutil
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from lxml import html, etree


print("----Airline delays data----")

if not os.path.exists('airline_data.csv'):
    print("airline_data.csv not found, constructing")
    if not os.path.exists("airline_files.log"):
        print("""
        Unfortunately I can't download this data automatically :(

        To obtain the data:
        - visit https://www.transtats.bts.gov/,
        - go to the aviation section,
        - go to Airline On-Time Performance Data,
        - go to Reporting Carrier data from 1987 to present,
        - click "download", and then for whichever time period, download the following fields:

        -----
        Year
        Month
        DayOfMonth
        DayOfWeek
        Reporting_Airline
        OriginAirportID
        Origin
        OriginCityName
        OriginState
        DestAirportID
        Dest
        DestCityName
        DestState
        CRSDepTime
        DepTime
        DepDelay
        TaxiOut
        WheelsOff
        WheelsOn
        TaxiIn
        CRSArrTime
        ArrDelay
        Cancelled
        Diverted
        CRSElapsedTime
        ActualElapsedTime
        AirTime
        Distance
        CarrierDelay
        WeatherDelay
        NASDelay
        SecurityDelay
        LateAircraftDelay
        -----

        Then create a file called "airline_files.log" here, with the filename of each zip file you downloaded, one name per line.

        Then run this script again.
        """)
        quit()

    print("loading zip file manifest in airline_files.log")
    with open("airline_files.log", "r") as f:
        zipfns = f.readlines()

    print("unzipping files")
    csv_filenames = []
    for line in zipfns:
        fn = line.strip()
        print(f"unzipping file: {fn}")
        out_csv_name = fn[:-3]+'csv'
        csv_filenames.append(out_csv_name)
        if os.path.exists(out_csv_name):
            print(f"{out_csv_name} already exists, skipping")
            continue
        # Unzip the geocoding data
        with zipfile.ZipFile(fn, "r") as zf:
            zinfo = zf.infolist()
            for zi in zinfo:
                zi.filename = out_csv_name
                zf.extract(zi)

    print("combining into one csv file")
    for i in range(len(csv_filenames)):
        fn = csv_filenames[i]
        with open(fn, "r") as source, open("airline_data.csv", "a") as dest:
            if i > 0:
                source.readline() # remove the first header line from the file for all but the first file
            shutil.copyfileobj(source, dest)
else:
    print("airline_data.csv exists, loading")

df = pd.read_csv('airline_data.csv')

airport_codes = ['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'SFO', 'SEA']
weather_codes = ['K'+code for code in airport_codes]

print('Filtering data to origin airport {airport_codes}')
df = df[df.ORIGIN.isin(airport_codes)]

print('Collecting unique dates')
df_weather = df[['YEAR', 'MONTH', 'DAY_OF_MONTH']].drop_duplicates().reset_index(drop=True)
print(f'Found {df_weather.shape[0]} unique dates')

print('Collecting weather information for each date & airport')
features = ['Mean Temperature', 'Max Temperature', 'Min Temperature', 'Dew Point', 'Average Humidity', 'Maximum Humidity', 'Minimum Humidity', 'Precipitation', 'Snow', 'Snow Depth', 'Sea Level Pressure', 'Wind Speed', 'Max Wind Speed', 'Visibility']
features = [f.lower().replace(' ', '') for f in features]
if not os.path.isdir('weather_data'):
    os.mkdir('weather_data')
for wcode in weather_codes:
    for i in range(df_weather.shape[0]):
        date_string = f"{df_weather.iloc[i].YEAR}-{df_weather.iloc[i].MONTH}-{df_weather.iloc[i].DAY_OF_MONTH}"
        html_path = 'weather_data/'+wcode+'-'+date_string+'.html'
        print(f"Checking if weather data {html_path} exists")
        # get the html file
        if not os.path.exists(html_path):
            print(f"Doesn't exist, downloading")
            req_prefix = 'https://www.wunderground.com/history/daily/'
            req_url = req_prefix + wcode +'/date/' + date_string
            #resp = requests.get()
            options = webdriver.ChromeOptions()
            options.add_argument('headless')
            options.add_argument('--enable-javascript')
            #user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36"
            #options.add_argument(f"user-agent={user_agent}")
            driver = webdriver.Chrome(options=options)
            driver.get(req_url)
            WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, '//td[@class="ng-star-inserted"]')))
            html = driver.page_source
            with open(html_path, 'w') as f:
                f.write(html)
            driver.quit()
            time.sleep(30)
        # read the csv
        df = pd.read_html(html_path)
quit()

apt_weather_data = np.empty((apt_day_data.shape[0], len(features)))
#set defaults for values if missing
apt_weather_data[:] = np.nan
apt_weather_data[:, 8] = 0.
apt_weather_data[:, 9] = 0.
apt_weather_data[:, 11] = 0.
for i in range(apt_day_data.shape[0]):
  year = str(apt_day_data[i, 0])
  month = str(apt_day_data[i, 1])
  day = str(apt_day_data[i, 2])
  print('On ' + year+'/'+month+'/'+day)
  table = np.array(pd.read_html(req_prefix+'/'+weather_code+'/'+year+'/'+month+'/'+day+'/DailyHistory.html')[0])[:, :2]
  names = []
  vals = []
  for j in range(table.shape[0]):
    names.append(str(table[j, 0].encode('utf-8').decode('ascii', 'ignore')))
    try:
      vals.append(float(non_decimal.sub('', table[j,1].encode('utf-8').decode('ascii', 'ignore'))))
    except:
      vals.append(np.nan)
  vals = np.array(vals)
  names = [nm for j, nm in enumerate(names) if not np.isnan(vals[j])]
  names = [nm.lower().replace(' ', '') for nm in names]
  vals = vals[np.isnan(vals) == 0]
  for j, nm in enumerate(names):
    try:
      apt_weather_data[i, features.index(nm)] = vals[j]
    except:
      pass
year_weather_data.append(apt_weather_data)
year_day_data.append(apt_day_data)

print('done retrieving and parsing, saving output')
weather_data = np.vstack(year_weather_data)
day_data = np.vstack(year_day_data)
X = np.zeros((weather_data.shape[0], weather_data.shape[1]+2))
X[:, :weather_data.shape[1]] = weather_data
for i in range(weather_data.shape[0]):
  X[i, weather_data.shape[1]] = (dt.date(day_data[i, 0], day_data[i, 1], day_data[i, 2]) - dt.date(1987, 1, 1)).total_seconds()/(60.*60.*24.*365.)
X[:, weather_data.shape[1]] = X[:, weather_data.shape[1]] - X[:, weather_data.shape[1]].min()
X[:, -1] = 1.
y = day_data[:, -1]
inds = np.logical_not(np.any(np.isnan(X), axis=1))
X = X[inds, :]
y = y[inds]

inds = np.arange(X.shape[0])
np.random.shuffle(inds)
n_train = 9*X.shape[0]/10
Xt = X[inds[n_train:], :]
yt = y[inds[n_train:]]
X = X[inds[:n_train], :]
y = y[inds[:n_train]]

np.savez('data/airportdelays.npz', X=X, y=y, Xt=Xt, yt=yt)




