# -*- coding: utf-8 -*-
"""
Basic Functions for bachelor project

Created on Thu Aug 30 13:37:46 2018

@author: Jakob Løfgren
"""

import numpy as np
import pandas as pd
import os.path
import sys


class PC:
    """Object to identify which pc is running 
    
    Identifies the system the code is run on, and sets up the paths
    Stops execution on unknown system

    Attributes
    ----------
    
    PC_running : int
        Number used to define PC (1 = laptop, 2 = stationary)
    
    PC_string : str
        String defining PC
        
    DataPath : str
        String defining where data can be found
        
    GraphPath : str
        String defining where graphs should be saved
        
    """
    
    def __init__(self):
        """Initialization of object
        
        Initializes the object by figuring out the system 
        the code is running on
        """
        if os.path.isdir('C:/Users/JakobLab'):
            self.PC_Running = 1
            self.PC_String = 'Laptop'
        elif os.path.isdir('C:/Users/Jakob'):
            self.PC_Running = 2
            self.PC_String = 'Stationary'
        else:
            self.PC_Running = 0
            self.PC_String = 'Unkown'
        print('Running script on: ',self.PC_String)
        self.UpdatePath()
    def UpdatePath(self):
        """Updates path
        
        Creates the paths for data and graphs, depending on system 
        """
        if self.PC_Running == 1:
            self.DataPath = 'C:/Users/JakobLab/Documents/GitKraken/Bachelor Project/bachelor-project/Ignore/Data/'
            self.GraphPath = 'C:/Users/JakobLab/Documents/GitKraken/Bachelor Project/bachelor-project/Graphs/'
            self.MplStyle = 'C:/Users/JakobLab/.matplotlib/matplotlibrc/mystyle.mplstyle'
        elif self.PC_Running == 2:
            self.DataPath = 'C:/Users/Jakob/Documents/Bachelor Project/Bachelor Project/Ignore/Data/'
            self.GraphPath = 'C:/Users/Jakob/Documents/Bachelor Project/Bachelor Project/Graphs/'
            self.MplStyle = 'C:/Users/Jakob/.matplotlib/matplotlibrc/mystyle.mplstyle'
        else:
            self.DataPath = ''
            self.GraphPath = ''
            self.MplStyle = ''
            sys.exit('System Unkown, Stopping execution')


def ImportSevData(DataPath, FileName):
    """Reads .csv file
    
    Reads the csv file defined by `FileName` from `DataPath`
    then pickles it for later read
    
    Parameters
    ----------
    DataPath : str
        Complete path to datafiles
        
    FileName : str
        Name of csv to read
        
    Returns
    -------
    Sev_Middel_Data : array-like
        A dataframe containing the energy production data
        
    Notes
    -----
    File should be comma seperated, one line of headers
    
    Columns are `Time` , `Diesel Mid` , `Water Mid` , `Wind Mid` , `Diesel Max` , `Water Max` , `Wind Max` , `Diesel Min` , `Water Min` , `Wind Min`
    
    parses dates, day first
    """
    print('SEV file does not exist, Reading from csv...')
    #Read the sev data into seperate dataframes
    Sev_Middel_Data = pd.read_csv(DataPath+FileName+'.csv',skiprows=1,
                                  header=None, index_col=0, usecols=[0,1,2,3],
                                  parse_dates=True, dayfirst=True, 
                                  names=['Time','Diesel','Water','Wind'])
     
    #After having seen the data, it is seen that 
    #certain values are outliers, these are NaN'ed
    Sev_Middel_Data = Sev_Middel_Data.mask(Sev_Middel_Data.sub(Sev_Middel_Data.mean()).div(Sev_Middel_Data.std()).abs().gt(4))             
    #Create a total column
    Sev_Middel_Data['Total'] = Sev_Middel_Data.sum(axis = 1)
  
    #Save pickle for loading later
    Sev_Middel_Data.to_pickle(DataPath+'Sev_Mid.pkl')
    return Sev_Middel_Data

def ImportWeatherDataOWM(DataPath, FileName):
    """Reads .csv file from OpenWeatherMap
    
    DEPRECATED
    
    Reads the csv file defined by `FileName` from `DataPath`
    then pickles it for late use
    
    Parameters
    ----------
    DataPath : str
        Complete path to datafiles
        
    FileName : str
        Name of csv to read
        
    Returns
    -------
    Weather_Data : array-like
        A dataframe containing the weather data
        
    Notes
    -----
    File should be comma seperated, one line of headers
    
    Columns are `Time` , `Temperature` , `Temp Min` , `Temp Max` , `Pressure` , `Humidity` , `Wind speed` , `Wind direction` , `Cloud coverage`
    
    parses dates, day first
    """
    print('Weather file does not exist, Reading from csv...')
    #Read the sev data into seperate dataframes
    Weather_Data = pd.read_csv(DataPath+FileName+'.csv',skiprows=1,
                               header=None,index_col=0,
                               usecols=[1,6,7,8,9,12,13,14,23],
                               names=['Time','Temp','Temp_Min','Temp_Max',
                                      'Pressure','Humidity','Wind_speed',
                                      'Wind_Dir','Clouds'],
                               parse_dates=True,dayfirst=True)
    Weather_Data = Weather_Data[~Weather_Data.index.duplicated(keep='last')]

    Weather_Data = Weather_Data.mask(Weather_Data.sub(Weather_Data.mean()).div(Weather_Data.std()).abs().gt(4))
    
    Weather_Data['Temp_C'] = Weather_Data['Temp']-272.15
    Weather_Data['Temp_Max_C'] = Weather_Data['Temp_Max']-272.15
    Weather_Data['Temp_Min_C'] = Weather_Data['Temp_Min']-272.15
  
    
    #Save pickle for loading later
    Weather_Data.to_pickle(DataPath+'Weather.pkl')
    return Weather_Data

def ImportWeatherData(DataPath, FileName):
    """Reads .csv file from MeteoBlue
    
    Reads the csv file defined by `FileName` from `DataPath`
    then pickles it for late use
    
    Parameters
    ----------
    DataPath : str
        Complete path to datafiles
        
    FileName : str
        Name of csv to read
        
    Returns
    -------
    Weather_Data : array-like
        A dataframe containing the weather data
        
    Notes
    -----
    Reads entire CSV, but only keeps the relavant years
    
    CSV has 12 rows of explanation
    
    Semicolon seperated
    

    Temperature (2m) and relative humidity (2m): Comparable to measurements at 2 meters above ground.
    Pressure: Atmospheric air pressure reduced to mean sea level as most commonly used for weather reports. The local pressure varies with altitude. Locations at higher elevation have a lower local atmospheric pressure.
    Precipitation amount: Total precipitation amount including rain, convective precipitation and snow. 1mm at 10:00 is comparable to a rain-gauge measurement from 9:00-10:00.
    Snowfall amount: Fraction of total precipitation that falls down as snow and is converted to cm instead of mm.
    Total cloud cover: Percentage of the sky that is covered with clouds: 50% means half of the sky is covered. 0-25% clear sky, 25-50% partly cloudy, 50-85% mostly cloudy and above 85% overcast.
    Low, mid and high cloud cover: Cloud cover at different altitudes. High clouds (8-15 km) like cirrus are less significant for total cloud cover than low (below 4 km) like stratus, cumulus and fog or mid clouds (4-8 km) like alto cumulus and alto stratus.
    Solar radiation: Global radiation (diffuse and direct) on a horizontal plane given in Watt per square meter.
    Wind speed: Hourly average wind speeds at given altitude levels "10 and 80 meters above ground" or pressure level "900 hPa". Units can be selected.
    Wind direction: Wind direction in degrees seamless from 0° (wind blowing from north), 90° (east wind), 180° (south wind) and 270° (west wind).
    Wind gusts: Short term wind speed turbulence in an hour. Gusts indicate the level of turbulence as such they could be lower than regular wind speeds.
    """
    
    print('Weather file does not exist, Reading from csv...')
    #Read the sev data into seperate dataframes
    Weather_Data = pd.read_csv(DataPath+FileName+'.csv',skiprows=12,sep=';',
                               names=['Year','Month','Day','Hour','Minute',
                                      'Temp','Hum','SLP','Precip','Rain',
                                      'Cloud','Cloud_high','Cloud_mid',
                                      'Cloud_low', 'Sun_dur','Rad',
                                      'W_speed_10','W_dir_10',
                                      'W_speed_80','W_dir_80',
                                      'W_speed_900','W_dir_900',
                                      'Gust'])
    Weather_Data = Weather_Data.set_index(pd.to_datetime(Weather_Data.loc[:,'Year':'Minute']))
    
    #Weather_Data = Weather_Data[~Weather_Data.index.duplicated(keep='last')]

    #Weather_Data = Weather_Data.mask(Weather_Data.sub(Weather_Data.mean()).div(Weather_Data.std()).abs().gt(2))
    
    Weather_Data = Weather_Data.loc['2013-03-05 00-00':'2018-03-31 23-00']
    #Save pickle for loading later
    Weather_Data.to_pickle(DataPath+'Weather.pkl')
    return Weather_Data


def CombineData(DataPath, Sev_Middel_Data, Weather_Data,save=True):
    """Combines the energy and weather data
    
    Combines both the energy and weather data
    
    Parameters
    ----------
    DataPath : str
        Complete path to datafiles
        
    Sev_Middel_Data : array-like
        Energy production data
        
    Weather_Data : array-like
        Weather Data
        
    save : bool , optional
        Chooses whether to pickle the data
        
    Returns
    -------
    Data : array-like
        A dataframe containing the combined data
        
    Notes
    -----
    Merges using the left outer join
    """
    print('Combined data file does not exist')
    Data = pd.merge(Sev_Middel_Data,Weather_Data,how='left',left_index=True,
                    right_index=True)
    if save:
        Data.to_pickle(DataPath+'Combined.pkl')
    return Data

def RemNaN(Data):
    """Remove NaN
    
    Remove undefined values, removes entire rows
    
    Parameters
    ----------
    Data : array-like
        Data to remove NaNs from
    
    Returns
    -------
    Data : array-like
        A dataframe without NaN
    """
    return Data.dropna(axis='index',how='any')

def StandardizeData(Data, std_Data=None):
    """Standardizes the data
    
    Z-score standardizes the data
    optionally by other datasets mean and standard deviation
    
    Parameters
    ----------
    Data : array-like
        The data to standardize
        
    std_Data : array-like , optional
        An optional parameter to standardize by
        
    Returns
    -------
    stdized_Data : array-like
        The data standardized
        
    Notes
    -----
    Basic Z-score standardization
    X - X_mean / X_std
    
    """
    if std_Data is None:
        std_Data = Data
    stdized_data = (Data - std_Data.mean()) / std_Data.std()
    return stdized_data

def MakePercent(data):
    data = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))*100
    return data

def MakePercentTotal(data, Min=None, Max=None):
    if Min is None:
        Min = min(data.Total)
    if Max is None:
        Max = max(data.Total)
          
    data = (data-Min)/(Max-Min)*100
    return data

def CreateCapData(data):
    data["max_wind"] = 5*0.9
    data["max_wind"].loc['2014-10-09':'2018-03-31'] = 18*0.9
    data["max_diesel"] = 65
    data["Max_water"] = 39
    return data
    
def readReadme(path):
    """Read readme in path
    
    Reads important information from the readme in path
    
    Parameters
    ----------
    path : str
        The path to the readme file
    
    Returns
    -------
    history : dict
        Dictionary containing full history
    """
    epoch = []
    infoline = ''
    loss = []
    val_loss = []
    mae = []
    val_mae = []
    history = {}
    found = False
    with open(path) as f:
        for line in f:
            if '*' in line:
                infoline = line.split('*')[1] 
            if line == '___\n':
                found = False
            if found:
                epoch.append(line.split('|')[1])
                loss.append(line.split('|')[2])
                val_loss.append(line.split('|')[3])
                mae.append(line.split('|')[4])
                val_mae.append(line.split('|')[5])
                
            if line == '### Training history ### \n':
                found = True
    try:        
        history[epoch[0].replace(' ','')] = [int(x) for x in epoch[2:]]
        history[loss[0].replace(' ','')] = [float(x) for x in loss[2:]]
        history[val_loss[0].replace(' ','')] = [float(x) for x in val_loss[2:]]
        history[mae[0].replace(' ','')] = [float(x) for x in mae[2:]]
        history[val_mae[0].replace(' ','')] = [float(x) for x in val_mae[2:]]
    except ValueError:
        print('')
        print('Erroneous history format in: ')
        print(path)
    except IndexError:
        print('')
        print('No Data in: ')
        print(path)
    return history, infoline
            
def readReadmeCV(path=''):
    """Read readme file in CV folder
    
    Iterates through each folder in cwd/tmp/CV/ to find readmes.
    Reads important information from each readme
    
    Returns
    -------
    history : dict
        Dictionary containing full history
    
    listOfErrors : array-like
        A dictionaries with information from readmes
        
    Notes
    -----
    listOfErrors mimmicks the output from CVTest functions
    """
    history = {}
    found = False
    Tests = {}
    listOfErrors = {}
    Iterations = {}
    test = 'unknown'
    testval = 0
    if path == '':
        path = os.getcwd().replace('\\','/')+'/tmp/CV'+path
    
    for sub_dir, dirs, files in os.walk(path): 
        if os.path.isfile(sub_dir.replace('\\','/')+'/readme.md'):
            
            with open(sub_dir.replace('\\','/')+'/'+files[0]) as f:
                epoch = []
                loss = []
                val_loss = []
                mae = []
                val_mae = []
                print(sub_dir.replace('\\','/')+'/'+files[0])
                #print(sub_dir.replace('\\','/').split('/')[-2])
                testval = 0
                for line in f:
                    if line == '___\n':
                        found = False
                    if '*' in line:
                        infoline = line.split(' ')
                        test = infoline[3]
                        if test == 'layers':
                            testval = infoline[4].split('=')[1].split(',')[0]
                        if test == 'nodes':
                            testval = int(infoline[5].split('=')[1])*256
                        if test == 'hyperparam':
                            testval = str(sub_dir.replace('\\','/').split('/')[-2])


                    if found:
                        epoch.append(line.split('|')[1])
                        loss.append(line.split('|')[2])
                        val_loss.append(line.split('|')[3])
                        mae.append(line.split('|')[4])
                        val_mae.append(line.split('|')[5])

                    if line == '### Training history ### \n':
                        found = True
                #print('run'+str(sub_dir.split('\\')[-1]))
                Iterations['run'+str(sub_dir.split('\\')[-1])]={'loss':float(loss[-1]),
                               'val_loss': float(val_loss[-1]),
                               'MAE': float(mae[-1]),
                               'val_MAE': float(val_mae[-1]),
                               'Epoch': int(epoch[-1])}
                #print(sub_dir.split('\\')[-1])
            #if sub_dir.split('\\')[-1] == '9':
            history['Epoch'] = epoch[2:]
            history['Loss'] = loss[2:]
            history['Val Loss'] = val_loss[2:]
            history['MAE'] = mae[2:]
            history['Val MAE'] = val_mae[2:]
        else:
                #print('hej\n')
            Tests[test+str(testval)] = Iterations
            print(str(testval))
            Iterations = {}
            
        
    try:
        del Tests['unknown0']
    except:
        pass
    listOfErrors = {'runs': Tests}
    return history, listOfErrors

def readReadmeFolder():
    """Read readme files in the entire tmp folder
    
    Iterates through each folder in cwd/tmp/ to find readmes.
    Reads important information from each readme
    
    Returns
    -------
    list_of_runs : array-like
        A list of dictionaries with information from readmes
    """
    list_of_runs = []
    temp_dict = {}
    run_dict = {}
    i = ''
    path = os.getcwd().replace('\\','/')+'/tmp/'
    for sub_dir, dirs, files in os.walk(path):
        #print('test')
        if not 'CV' in sub_dir:
            temp_dict['name'] = sub_dir.split('/')[-1]
            temp_dict['files'] = files if files else 'empty' 
            if 'readme.md' in files:
                h,i = readReadme(sub_dir.replace('\\','/')+'/readme.md')
                h_head = [*h]
                #print('h is:' )
                #print(h)
                #print('with header:' )
                #print(h_header)
                for n in range(len(h_head)):
                    run_dict[h_head[n]] = h[h_head[n]][-1] if h[h_head[n]] else []
            temp_dict['info'] = i if i else 'No description'
            temp_dict['train'] = run_dict if run_dict else 'No training'
            #print(temp_dict)
            list_of_runs.append(temp_dict)
            temp_dict = {}
            run_dict = {}
        else:
            if sub_dir.split('/')[-1] == 'CV':
                list_of_runs.append({'name': sub_dir.split('/')[-1],
                                     'info': 'Cross Validation',
                                     'dirs': dirs})
    return list_of_runs
        
            
        
    