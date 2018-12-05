# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:35:20 2018

Bachelor Project

Data driven estimation of diesel reserve for
renewable energy production on the Faroe Islands

DTU
@author: Jakob Løfgren S143225
"""

print('')
print('Running Bachelor Project')
print('Data driven estimation of diesel reserve for renewable energy production'
      +'on the Faroe Islands')
print('')


"""import"""
import pandas as pd
import os.path
import Functions as fcn #Functions created by me
import Graphing as grph #Graphing is done in here
import scipy.stats as stats
import MachineLearning as ML
import datetime as datetime
from tensorflow import keras


"""
Check which PC is running the script, to setup the correct paths
"""    
PC = fcn.PC()


"""
Read pickle files with data if they exist.
Otherwise read csv files and pickle them.
Finally combine Grid and Weather data.
"""
#Read the pickle files or create them
if os.path.isfile(PC.DataPath+'Sev_Mid.pkl'):
    print('SEV pickle found, reading from this')
    Sev_Middel_Data = pd.read_pickle(PC.DataPath+'Sev_Mid.pkl')
else:
    Sev_Middel_Data = fcn.ImportSevData(PC.DataPath, 'Sev')
    
if os.path.isfile(PC.DataPath+'Weather.pkl'):
    print('Weather pickle found, reading from this')
    Weather_Data = pd.read_pickle(PC.DataPath+'Weather.pkl')
else:
    Weather_Data = fcn.ImportWeatherData(PC.DataPath, 'Torshavn_Full')

if os.path.isfile(PC.DataPath+'Combined.pkl'):
    print('Combined pickle found, reading from this')
    Data = pd.read_pickle(PC.DataPath+'Combined.pkl')   
else:
    Data = fcn.CombineData(PC.DataPath, Sev_Middel_Data, Weather_Data)

MaxTot = max(Sev_Middel_Data.Total)
MinTot = min(Sev_Middel_Data.Total)




Data_percentTot = fcn.MakePercentTotal(Sev_Middel_Data.loc[:,'Diesel':'Total'],MinTot,MaxTot)
Data_percentTot = fcn.CombineData(None, Data_percentTot, Weather_Data,save=False)
Data_percentTot = fcn.RemNaN(Data_percentTot)

Data_percent = fcn.MakePercent(Data.loc[:,'Diesel':'Total'])
Data_percent = fcn.CombineData(None, Data_percent, Weather_Data,save=False)
Data_percent = fcn.RemNaN(Data_percent)

#Data = Data.drop(['W_speed_10','W_dir_10','W_speed_900','W_dir_900','Cloud_high',
#           'Cloud_mid','Cloud_low'],axis=1)
#Data = fcn.CreateCapData(Data)


MaxTot = max(Sev_Middel_Data.Total)
MinTot = min(Sev_Middel_Data.Total)


"""
#Add noise

windsd=2.2
windmean=18/2

dirsd=3
dirmean=0

tempsd=0.25
tempmean=1.1

dirnoise = pd.DataFrame(np.random.normal(dirmean, dirsd, [len(Data.Temp),1])) 
windnoise = pd.DataFrame(np.random.normal(windmean, windsd, [len(Data.Temp),1]))
tempnoise = pd.DataFrame(np.random.normal(tempmean, tempsd, [len(Data.Temp),1]))


dirnoise.to_csv(path_or_buf=os.getcwd().replace('\\','/')+'/dir.csv',sep=';',decimal=',')
windnoise.to_csv(path_or_buf=os.getcwd().replace('\\','/')+'/wind.csv',sep=';',decimal=',')
tempnoise.to_csv(path_or_buf=os.getcwd().replace('\\','/')+'/temp.csv',sep=';',decimal=',')
Data.to_csv(path_or_buf=os.getcwd().replace('\\','/')+'/data2.csv',sep=';',decimal=',')

data_noise = pd.read_csv(os.getcwd().replace('\\','/')+'/data2.csv',
                         header=0,index_col=0,
usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                parse_dates=True,dayfirst=True,delimiter=';',decimal=b',')

Trainn_label, Trainn_data, Testn_label, Testn_data = ML.Split(data_noise)

Testn_data = fcn.StandardizeData(Test_data, std_Data=Train_data).fillna(0)  
Trainn_data = fcn.StandardizeData(Train_data, std_Data=Train_data).fillna(0)

Testn_data=Testn_data.drop(axis=1,labels=['W_speed_80','W_dir_80','W_speed_900','W_dir_900'])

pred = ML.Predict(Model,Testn_data)

off = pred-Testn_label
abs(off).sum()/len(off.Diesel)
"""



"""
Create basic graphs, to get a feel for the data
Will take a while to run, as a 27x27 scatter plot is created
Only creates graphs if they arent already saved
"""
grph.BasicGraphs(PC.GraphPath, Data_percentTot)
#grph.WeatherGraphs(PC.GraphPath, Weather_Data)
if not os.path.isfile(PC.GraphPath+'ScatterPlot.png'):
    print('Scatterplot not existing, creating it')
    grph.ScatterMatrix(PC.GraphPath, Data_percent, 'ScatterPlot')


#Clean NaN's from the dataset
print("NaN's in data set before cleaning: "
      + str(Data.isnull().sum().sum()))
Data_NoNAN = fcn.RemNaN(Data)
print("NaN's in data set after cleaning: "
      + str(Data_NoNAN.isnull().sum().sum()))

#Standardization of the dataset
Data_Stand = fcn.StandardizeData(Data_NoNAN)
print("NaN's in data set after standardization: "
      + str(Data_Stand.isnull().sum().sum()))
# Here it is the minute column that gets NA
# As it divides by zero
# solution: set entire column to zero again
Data_Stand = Data_Stand.fillna(0)
print("NaN's in data set after standardization: "
      + str(Data_Stand.isnull().sum().sum()))

Data_NoNAN = fcn.CreateCapData(Data_NoNAN)


if not os.path.isfile(PC.GraphPath+'ScatterPlot_Norm.png'):
    print('Normalized scatterplot not existing, creating it')
    grph.ScatterMatrix(PC.GraphPath, Data_Stand, 'ScatterPlot_Norm')



Data_NoNAN.describe()

#if (not os.path.isfile(PC.DataPath+'Trainlab.pkl') or
#    not os.path.isfile(PC.DataPath+'Traindat.pkl') or
#    not os.path.isfile(PC.DataPath+'Testdat.pkl') or
#    not os.path.isfile(PC.DataPath+'Testdat.pkl')):
Train_label, Train_data, Test_label, Test_data = ML.Split(Data_NoNAN)
#    Train_label.to_pickle(PC.DataPath+'Trainlab.pkl')
#    Train_data.to_pickle(PC.DataPath+'Traindat.pkl')
#    Test_label.to_pickle(PC.DataPath+'Testlab.pkl')
#    Test_data.to_pickle(PC.DataPath+'Testdat.pkl')
#else:
#    Train_label = pd.read_pickle(PC.DataPath+'Trainlab.pkl')
#    Train_data = pd.read_pickle(PC.DataPath+'Traindat.pkl')
#    Test_label = pd.read_pickle(PC.DataPath+'Testlab.pkl')
#    Test_data = pd.read_pickle(PC.DataPath+'Testdat.pkl')

# Standardize the data based on mean and std dev of train data
# Standardize test data first, so that it doesn't standardize
# on already standardized data

Test_data = fcn.StandardizeData(Test_data, std_Data=Train_data).fillna(0)  
Train_data = fcn.StandardizeData(Train_data, std_Data=Train_data).fillna(0)

Train_data = Train_data.drop(axis=1,labels=['W_speed_10','W_dir_10','W_speed_900','W_dir_900','Cloud_high','Cloud_mid','Cloud_low'])
Test_data = Test_data.drop(axis=1,labels=['W_speed_10','W_dir_10','W_speed_900','W_dir_900','Cloud_high','Cloud_mid','Cloud_low'])

def MakeModel():
    checkpoint_path='/tmp/'+str(datetime.datetime.now())
    
    optimizer = keras.optimizers.Adam(lr=0.0006, beta_1=0.96, 
                                  beta_2=0.99999, epsilon=1e-2)
    
    descriptor = 'Final training of model'
    filefmt = 'weights.Epoch-{epoch:03d};Loss-{val_loss:.6f}.hdf5'
    
    
    #Model = ML.SimpleModel(Train_data,optimizer)
    Model = ML.ModularModel(Train_data, optimizer, layers=4, nodes=4*256)
    
    History = ML.TrainModel(Model, Train_data, Train_label,
                            EPOCHS=500, min_delta=0.0, patience=20, PERIOD=0,
                            BATCH=45, val_data=tuple([Test_data,Test_label]),
                            checkpoint_path=checkpoint_path,file_name=filefmt,
                            Descriptor=descriptor)
    
    ML.PlotHistory(History,save_path= checkpoint_path.replace('.',':').replace(':','-')+'/')
    Predictions = ML.Predict(Model, Test_data)
    grph.PlotHistory(Predictions, Test_label,  os.getcwd().replace('\\','/')+checkpoint_path.replace('.',':').replace(':','-')+'/')
    grph.PlotHistory2018(Predictions, Test_label,  os.getcwd().replace('\\','/')+checkpoint_path.replace('.',':').replace(':','-')+'/')
    grph.PlotHistoryDiff(Predictions, Test_label,  os.getcwd().replace('\\','/')+checkpoint_path.replace('.',':').replace(':','-')+'/')
    grph.PlotHistory2018percent(Predictions, Test_label,  os.getcwd().replace('\\','/')+checkpoint_path.replace('.',':').replace(':','-')+'/')
    
    Offset = Predictions - Test_label
    OffsetP = Offset / Test_label * 100
    return History, Model, OffsetP, Offset

def CVTest(test='layers', start=4, finish=14,
           optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.95, 
                                      beta_2=0.999, epsilon=1e-4)):
    
    val_labels = Data_NoNAN.loc[:, 'Diesel':'Total']
    val_data = fcn.StandardizeData(Data_NoNAN.loc[:, 'Year':'Gust']).fillna(0)
    
    
    listOfErrors = ML.CrossValidation(data=val_data, labels=val_labels,
                                      test=test, start=start, finish=finish,
                                      optimizer=optimizer)
    return listOfErrors

def goThroughReadme(folder):
    path = os.getcwd().replace('\\','/')+'/tmp/'+folder+'/readme.md'
    readme, __= fcn.readReadme(path)
    fig = grph.plotFromReadme(readme, [*readme][1:]) 
    return readme, fig
            
""" Funky things to do
#Get wights of layer 'name'
Model.get_layer(name='Output').get_weights()

#Get config of optimizer
Model.optimizer.get_config()

#Load model
Model = keras.models.load_model((os.getcwd().replace('\\','/')
                                +'/final/final.hdf5'),
                                custom_objects={'my_loss': ML.my_loss})


#Save list of errors
with open(os.getcwd().replace('\\','/')
            +'/tmp/CV/listOfErrors.dill', 'wb') as f:
    dill.dump(ls,f)

#Load list of errors
with open(os.getcwd().replace('\\','/')
            +'/tmp/CV/listOfErrors.dill', 'rb') as f:
    ls = dill.load(f)
    
#Find the best loss in readmefolder output
vm = []
for n in range(len(__)):
    try:
        vm.append(__[n]['train']['ValMAE']) if __[n]['train']['ValMAE'] else None 
    except KeyError:
        pass
    except TypeError:
        pass
print(vm)
min(vm)


# Run CV test on hyperparameters
y=0
rangevar = [0.99,0.999,0.9999,1]
#for k in range(10):
#    rangevar.append(random.uniform(0.0, 0.999))
for x in range(4):
    y+=1
    lr=rangevar[x]
    print(lr)
    optimizer = keras.optimizers.Adam(lr=0.0006, beta_1=lr, 
                                  beta_2=0.999, epsilon=1e-2)
    ls = CVTest('hyperparam', y, y+1, optimizer)
    
y=10
rangevar = [0.99,0.999,0.9999,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98]
#for k in range(10):
rangevar=[0.9999999, 1.0,0.99999999]
for x in range(3):
    y+=1
    lr=rangevar[x]
    print(lr)
    optimizer = keras.optimizers.Adam(lr=0.0006, beta_1=0.96, 
                                  beta_2=0, epsilon=1e-2)
    ls = CVTest('hyperparam', y, y+1, optimizer)



h,l = fcn.readReadmeCV('C:/Users/JakobLab/Documents/GitKraken/Bachelor Project/bachelor-project/Python/tmp/CV/Layers')

rv=[0,1,2,3,4,5,6,7,8]
log = False
r,f = grph.CVGraph(l,'layers',0,PC.GraphPath,rangevar=rv,name='Layers',maxtot=MaxTot,log=log)
f.show()
r,f = grph.CVGraph(l,'hyperparam',0,PC.GraphPath,[0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.999,0.9999],'Beta1')
"""

def WndScatter():
    ydata = Data_percentTot.Wind
    xdata = [[Data_percentTot.W_speed_10,
              Data_percentTot.W_speed_80,
              Data_percentTot.W_speed_900]
            ,[Data_percentTot.W_dir_10,
              Data_percentTot.W_dir_80,
              Data_percentTot.W_dir_900]]
    path = PC.GraphPath
    row = 2
    columns = 3
    ylabel = 'Wind Energy [%]'
    xlabels = [['Wind speed 10m [km/h]',
                'Wind speed 80m [km/h]',
                'Wind speed 900mb [km/h]'],
               ['Wind direction 10m [deg]',
               'Wind direction 80m [deg]',
               'Wind direction 900mb [deg]']]
    xlim = [[1,170],[1,370]]
    title = 'Wind energy scatter plot'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

def CldScatter():
    ydata = Data_percentTot.Total
    xdata = [Data_percentTot.Cloud_high,
              Data_percentTot.Cloud_mid,
              Data_percentTot.Cloud_low,
              Data_percentTot.Cloud]
    path = PC.GraphPath
    row = 1
    columns = 4
    ylabel = 'Total Energy [%]'
    xlabels = ['Cloud cover 8-15 km [%]',
               'Cloud cover 4-8 km [%]',
               'Cloud cover < 4 km [%]',
               'Cloud cover [%]']
    xlim = [0,105]
    title = 'Cloud Cover scatter plot'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

def HourScatter():
    ydata = Data_percentTot.Hour
    xdata = [Data_percentTot.Diesel,
              Data_percentTot.Water,
              Data_percentTot.Wind,
              Data_percentTot.Total]
    path = PC.GraphPath
    row = 1
    columns = 4
    ylabel = 'Hour'
    xlabels = ['Diesel [%]',
               'Water [%]',
               'Wind [%]',
               'Total [%]']
    xlim = [0,105]
    title = 'Energy types by time of day'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

def MonthScatter():
    ydata = Data_percentTot.Month
    xdata = [Data_percentTot.Diesel,
              Data_percentTot.Water,
              Data_percentTot.Wind,
              Data_percentTot.Total]
    path = PC.GraphPath
    row = 1
    columns = 4
    ylabel = 'Month'
    xlabels = ['Diesel [%]',
               'Water [%]',
               'Wind [%]',
               'Total [%]']
    xlim = [0,105]
    title = 'Energy types by month'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

def WndTmpScatter():
    ydata = Data_percentTot.Wind
    xdata = [Data_percentTot.Temp]
    path = PC.GraphPath
    row = 1
    columns = 1
    ylabel = 'Wind Energy [%]'
    xlabels = ['Temp [C]']
    xlim = [-5,25]
    title = 'Wind energy vs. Temperature'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

def RadScatter():
    ydata = Data_percentTot.Total
    xdata = [Data_percentTot.Rad]
    path = PC.GraphPath
    row = 1
    columns = 1
    ylabel = 'Total Energy [%]'
    xlabels = ['Solar radiation [W/m2]']
    xlim = [-5,800]
    title = 'Total energy vs. Radiation'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

def RainPlot():
    ydata = Data_percentTot.W_speed_80
    xdata = [Data_percentTot.Temp]
    
    path = PC.GraphPath
    row = 1
    columns = 1
    ylabel = 'Precipitation [mm]'
    xlabels = ['Month']
    xlim = [[-7,20],[-7,20]]
    title = 'Wind energy scatter plot'
    
    f = grph.PlotScatter(ydata, xdata, path, row, columns,
                         ylabel, xlabels, xlim, title)
    return f

#Model.get_layer(name='Layer1').count_params()
#Model.get_layer(name='Layer1').get_config()



"""
Graphs of activation functions
import numpy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

z = np.arange(-5, 5, .1)
zero = np.zeros(len(z))
y = np.max([zero, z], axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, y)
ax.set_ylim([-2.0, 2.0])
ax.set_xlim([-5, 5])
ax.grid(True)
ax.set_xlabel('z')
ax.set_title('Rectified linear unit')

plt.show()


z = numpy.arange(-5, 5, .1)
sigma_fn = numpy.vectorize(lambda z: 1/(1+numpy.exp(-z)))
sigma = sigma_fn(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, sigma)
ax.set_ylim([-2.0, 2.0])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('z')
ax.set_title('sigmoid function')

plt.show()


z = np.arange(-5, 5, .1)
t = np.tanh(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, t)
ax.set_ylim([-2.0, 2.0])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('z')
ax.set_title('tanh function')

plt.show()
"""
print('Script complete')
