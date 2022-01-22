import pandas as pd
import numpy as np
import re
import os
import sys
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

df_airline = pd.read_csv('airline_data.csv')

airport_codes = ['ATL', 'LAX', 'ORD', 'DFW', 'JFK', 'SFO', 'SEA']
weather_codes = ['K'+code for code in airport_codes]

print('Filtering data to origin airport {airport_codes}')
df_airline = df_airline[df_airline.ORIGIN.isin(airport_codes)]
print('Resetting index')
df_airline.reset_index(inplace=True, drop=True)

print('Collecting unique dates')
df_dates = df_airline[['YEAR', 'MONTH', 'DAY_OF_MONTH']].drop_duplicates().reset_index(drop=True)
print(f'Found {df_dates.shape[0]} unique dates')

print('Collecting weather information for each date & airport')
features = ['Mean Temperature', 'Max Temperature', 'Min Temperature', 'Dew Point', 'Average Humidity', 'Maximum Humidity', 'Minimum Humidity', 'Precipitation', 'Snow', 'Snow Depth', 'Sea Level Pressure', 'Wind Speed', 'Max Wind Speed', 'Visibility']
features = [f.lower().replace(' ', '') for f in features]
if not os.path.isdir('weather_data'):
    os.mkdir('weather_data')
for wcode in weather_codes:
    for i in range(df_dates.shape[0]):
        date_string = f"{df_dates.iloc[i].YEAR}-{df_dates.iloc[i].MONTH}-{df_dates.iloc[i].DAY_OF_MONTH}"
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
            print("Done downloading, sleeping for 30s to be nice to the webserver (any quicker than that causes throttling)")
            time.sleep(30)

print('Constructing dataframe of historic weather data')
if not os.path.exists('historic_weather.csv'):
    print("The historic weather csv doesn't exist; building")
    df_weather = pd.DataFrame({ 'airport':[],
                            'year':[],
                            'month':[],
                            'day':[],
                            'temp_high_F':[],
                            'temp_low_F':[],
                            'temp_avg_F':[],
                            'hist_avg_temp_high_F':[],
                            'hist_avg_temp_low_F':[],
                            'hist_avg_temp_avg_F':[],
                            'precip_in':[],
                            'hist_avg_precip_in':[],
                            'dew_high_F':[],
                            'dew_low_F':[],
                            'dew_avg_F':[],
                            'max_wind_spd_mph':[],
                            'visibility':[],
                            'pressure_hg':[]})

    for wcode in weather_codes:
        print('')
        print(f"Processing airport {wcode}")
        for i in range(df_dates.shape[0]):
            date_string = f"{df_dates.iloc[i].YEAR}-{df_dates.iloc[i].MONTH}-{df_dates.iloc[i].DAY_OF_MONTH}"
            html_path = 'weather_data/'+wcode+'-'+date_string+'.html'
            sys.stdout.write(f"Row {i+1}/{df_dates.shape[0]}          \r")
            sys.stdout.flush()

            # read the html file
            df = pd.read_html(html_path)[0]
            # make sure it is in the expected coln format
            if (len(df.columns) != 5) or ('Temperature' not in df.columns[0][0]) or ('Actual' not in df.columns[1][0]) or ('Historic' not in df.columns[2][0]):
                assert False, f"dataframe column headers unexpected. \nncols {len(df.columns)} df:\n{df}"
            # remove unused columns
            df.columns = ['name', 'val', 'hist', 'unused', 'unused']
            df = df[['name', 'val', 'hist']]
            # remove unused rows
            df = df[:11]

            # make sure rows are in expected format
            if ("High Temp" not in df['name'][0]) or \
               ("Low Temp" not in df['name'][1]) or \
               ("Average Temp" not in df['name'][2]) or \
               ("Precipitation" not in df['name'][3]) or \
               ("Dew Point" not in df['name'][4]) or \
               ("High" not in df['name'][5]) or \
               ("Low" not in df['name'][6]) or \
               ("Average" not in df['name'][7]) or \
               ("Max Wind" not in df['name'][8]) or \
               ("Visibil" not in df['name'][9]) or \
               ("Sea Level" not in df['name'][10]):
                assert False, f"dataframe row names unexpected. \ndf:\n{df}"

            # gather row of data for the big dataframe
            row = {'airport':wcode,
                    'year':df_dates.iloc[i].YEAR,
                    'month':df_dates.iloc[i].MONTH,
                    'day':df_dates.iloc[i].DAY_OF_MONTH,
                    'temp_high_F':df['val'][0],
                    'temp_low_F':df['val'][1],
                    'temp_avg_F':df['val'][2],
                    'hist_avg_temp_high_F':df['hist'][0],
                    'hist_avg_temp_low_F':df['hist'][1],
                    'hist_avg_temp_avg_F':df['hist'][2],
                    'precip_in':df['val'][3],
                    'hist_avg_precip_in':df['hist'][3],
                    'dew_high_F':df['val'][5],
                    'dew_low_F':df['val'][6],
                    'dew_avg_F':df['val'][7],
                    'max_wind_spd_mph':df['val'][8],
                    'visibility':df['val'][9],
                    'pressure_hg':df['val'][10]}
            df_weather = df_weather.append(row, ignore_index=True)
    df_weather['year'] = df_weather['year'].astype(int)
    df_weather['month'] = df_weather['month'].astype(int)
    df_weather['day'] = df_weather['day'].astype(int)
    print("Saving historic weather dataframe")
    df_weather.to_csv("historic_weather.csv", index=False)

print('Historic weather data file exists, loading')
df_weather = pd.read_csv("historic_weather.csv")


print('Augmenting airline data with weather data')
# now we have df_airline with flight delay info, and df_weather with weather
# want to augment with the relevant airport weather conditions for each day
# first add dummy columns
newcols = ['temp_high_F', 'temp_low_F', 'temp_avg_F',
           'hist_avg_temp_high_F', 'hist_avg_temp_low_F', 'hist_avg_temp_avg_F',
           'precip_in', 'hist_avg_precip_in', 'dew_high_F', 'dew_low_F', 'dew_avg_F',
           'max_wind_spd_mph', 'visibility', 'pressure_hg']
df_airline[newcols] = 1.
# now fill these with correct values
for i in range(df_airline.shape[0]):
    yr = df_airline.iloc[i].YEAR
    mnth = df_airline.iloc[i].MONTH
    day = df_airline.iloc[i].DAY_OF_MONTH
    wcode = 'K'+df_airline.iloc[i].ORIGIN
    df_row = df_weather[(df_weather.year == yr) & (df_weather.month == mnth) & (df_weather.day == day) & (df_weather.airport == wcode)].reset_index()
    assert df_row.shape[0] == 1, "Found multiple matches..."
    df_airline.loc[i, newcols] = df_row.loc[0, newcols]

print('Done parsing data; saving')
df_airline.to_csv("airport_delays.csv", index=False)

#np.savez('data/airportdelays.npz', X=X, y=y, Xt=Xt, yt=yt)




