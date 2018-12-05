# -*- coding: utf-8 -*-
"""
Functions for visualizing data for Bachelor project

Created on Thu Aug 30 14:03:19 2018

@author: JakobLab
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os.path
import progressbar
import ProgBar

plt.ioff()
plt.style.use('C:/Users/Jakob/.matplotlib/matplotlibrc/mystyle.mplstyle')

monthDict={1:'Januar', 2:'Februar', 3:'Marts', 4:'April', 5:'Maj', 6:'Juni', 7:'Juli', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
Color1=(0,107/255,164/255) # light blue
Color2=(255/255,128/255,14/255) # light orange
Color3=(171/255,171/255,171/255) # gray
Color4=(89/255,89/255,89/255) # Dark gray
Color5=(95/255,158/255,209/255) # Teal
Color6=(200/255,82/255,0/255) # brown-orange
Color7=(137/255,137/255,137/255) # medium gray
Color8=(162/255,200/255,236/255) # very light blue
Color9=(255/255,188/255,121/255) # light orange
Color10=(207/255,207/255,207/255) # light gray

def BasicGraphs(GraphPath,Data):
    """Creates basic graphs
    
    Creates graphs of the energy production
    Only creates the graphs if they don't already exist
    
    Creates:
        Production graph of entire period
        Monthly and Daily graphs of production
    
    Parameters
    ----------
    GraphPath : str
        Path to save the graphs.
    Data : array-like
        The data to graph.
    """
    
    pcount=0
    toCreate = 0
    if not os.path.isfile(GraphPath+'Monthly/Share2018-3.png'):
        MonthGraphs = 60
    else:
        MonthGraphs = 0
    if not os.path.isdir(GraphPath+'Daily/'):
        os.makedirs(GraphPath+'Daily')
    if not os.listdir(GraphPath+'Daily'):    
        DayGraphs = 150
    else:
        DayGraphs = 0
    if not os.path.isfile(GraphPath+'Diesel.png'):
        toCreate += 1
    if not os.path.isfile(GraphPath+'Wind.png'):
        toCreate += 1
    if not os.path.isfile(GraphPath+'Water.png'):
        toCreate += 1
    if not os.path.isfile(GraphPath+'Total.png'):
        toCreate += 1
    toCreate += MonthGraphs
    toCreate += DayGraphs
    #toCreate += 1
    
    
    progbar = ProgBar.Progbar(target = toCreate, newline_on_end = False,
                            text_description='Creating overview graphs: ')
    
    
    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    plt.rcParams["figure.figsize"] = [16,9]
    
    
    if not os.path.isfile(GraphPath+'Diesel.png'):
        pcount+=1
        #print('Creating diesel overview graph')
        fig, ax = plt.subplots()    
        ax.plot(Data.index, Data.Diesel,color=Color1)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        # round to nearest years...
        datemin = np.datetime64(Data.index[0], 'Y')
        datemax = np.datetime64(Data.index[-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        plt.title('Timeseries')
        plt.xlabel('Time [date]')
        plt.ylabel('Diesel production [% of total]')
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.05, color ='black')
        fig.autofmt_xdate()
        plt.savefig(GraphPath+'Diesel.png')
        plt.close(fig)
        progbar.update(pcount)
    
    if not os.path.isfile(GraphPath+'Wind.png'):
        pcount+=1
        #print('Creating Wind overview graph')
        fig, ax = plt.subplots()    
        ax.plot(Data.index, Data.Wind,color=Color1)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        # round to nearest years...
        datemin = np.datetime64(Data.index[0], 'Y')
        datemax = np.datetime64(Data.index[-1], 'Y')\
                                       + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        plt.title('Timeseries')
        plt.xlabel('Time [date]')
        plt.ylabel('Wind production [% of total]')
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.05, color ='black')
        fig.autofmt_xdate()
        plt.savefig(GraphPath+'Wind.png')
        plt.close(fig)
        progbar.update(pcount)
        
    if not os.path.isfile(GraphPath+'Water.png'):
        pcount+=1
        #print('Creating Water overview graph')
        fig, ax = plt.subplots()    
        ax.plot(Data.index, Data.Water,color=Color1)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        # round to nearest years...
        datemin = np.datetime64(Data.index[0], 'Y')
        datemax = np.datetime64(Data.index[-1], 'Y')\
                                       + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        plt.title('Timeseries')
        plt.xlabel('Time [date]')
        plt.ylabel('Water production [% of total]')
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.05, color ='black')
        fig.autofmt_xdate()
        plt.savefig(GraphPath+'Water.png')
        plt.close(fig)
        progbar.update(pcount)
        
    if not os.path.isfile(GraphPath+'Total.png'):
        pcount+=1
        #print('Creating Total overview graph')
        fig, ax = plt.subplots()    
        ax.plot(Data.index, Data.Total,color=Color1)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        # round to nearest years...
        datemin = np.datetime64(Data.index[0], 'Y')
        datemax = np.datetime64(Data.index[-1], 'Y')\
                                       + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        plt.title('Timeseries')
        plt.xlabel('Time [date]')
        plt.ylabel('Total production [%]')
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.05, color ='black')
        fig.autofmt_xdate()
        plt.savefig(GraphPath+'Total.png')
        plt.close(fig)
        progbar.update(pcount)
    
   
    majorTick = mdates.DayLocator(interval=7)  # every month
    majorFmt = mdates.DateFormatter('%d/%m')
    minorTick = mdates.DayLocator() # every day
    if not os.path.isdir(GraphPath+'Monthly'):
        #print('Monthly folder not found, creating it')
        os.makedirs(GraphPath+'Monthly')
    if not os.path.isfile(GraphPath+'Monthly/Share2018-3.png'):
        progbar = ProgBar.Progbar(target = toCreate, newline_on_end = False,
                            text_description='Creating monthly graphs: ')
        month=3
        year=3
        done = False
        count = 0
        #print('Creating Monthly share graph [%]')
        while not done:
            
            pcount += 1
            #if count % 6 == 0:
            #    print('\u220E', end='', flush=True)
            StartPoint = Data.index.get_loc\
                ('201'+str(year)+'-'+str(month)+'-'+'01 00:00',method='nearest')
            EndPoint = Data['201'+str(year)+'-'+str(month)].shape[0]
            fig, ax = plt.subplots()  
            ax.plot(Data.index[StartPoint:StartPoint+EndPoint],
                    Data.Diesel['201'+str(year)+'-'+str(month)])
            # format the ticks
            ax.xaxis.set_major_locator(majorTick)
            ax.xaxis.set_major_formatter(majorFmt)
            ax.xaxis.set_minor_locator(minorTick)
            datemin = np.datetime64(Data.index[StartPoint],
                                    'm')# round to nearest years...
            datemax = np.datetime64(Data.index[StartPoint+EndPoint-1],
                                    'm')
            ax.set_xlim(datemin, datemax)
            ax.grid(which='major', alpha = 1, color='black')
            ax.grid(which='minor', alpha = 0.05, color ='black')
            fig.autofmt_xdate()
            Diesel = Data.Diesel['201'+str(year)+'-'+str(month)]
            Water = Data.Water['201'+str(year)+'-'+str(month)]+Diesel
            Wind = Data.Wind['201'+str(year)+'-'+str(month)]+Water
            plt.plot(Water,color=Color2)
            plt.plot(Wind,color=Color3)
            plt.fill_between(Wind.index,Wind, Water, facecolor=Color3, alpha=1)
            plt.fill_between(Water.index,Water, Diesel, facecolor=Color2, alpha=1)
            plt.fill_between(Diesel.index,Diesel, 0, facecolor=Color1, alpha=1)
            plt.title('201'+str(year)+' - '+monthDict[month])
            plt.ylim([-1,105])
            plt.xlabel('Time [date]')
            plt.ylabel('Total production [%]')
            plt.legend(['Diesel','Water','Wind'])
            plt.savefig(GraphPath+'Monthly/'+'Share201'+str(year)+'-'+str(month)+'.png')
            plt.close(fig)
            progbar.update(pcount)
            if month == 12:
                month = 1
                year = year + 1
            else:
                month = month+1
            if year == 8 and month == 4:
                done = True
                #print('')
                
    """
    If not created, create "random" graphs of single days
    This can help see the flow of a day
    """
    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    hours = mdates.HourLocator() # every hour
    hoursM = mdates.HourLocator(interval=3) # every 3rd hour
    yearsFmt = mdates.DateFormatter('%d')
    hoursFmt = mdates.DateFormatter('%H:%M')

    if not os.listdir(GraphPath+'Daily') :
        progbar = ProgBar.Progbar(target = toCreate, newline_on_end = False,
                            text_description='Creating daily graphs: ')
        #widgets = ['Creating daily share graph: ', progressbar.Percentage(),
        #           ' ',progressbar.Bar(marker='∎',left='[',right=']'),
        #           ' ', progressbar.AdaptiveETA()]
        #pbar = progressbar.ProgressBar(widgets=widgets, maxval=150)
        #pbar.start()
        #print('')
        #print('Creating Daily share graph [%]')
        for count in range(0,DayGraphs):
            #if count % 5 == 0:
            #    print('\u220E', end='', flush=True)
            
            #pbar.update(count+1)
            pcount +=1
            year=np.random.randint(3,7+1)
            if year == 3:
                month=np.random.randint(3,12+1)
            elif year == 8:
                month=np.random.randint(1,3+1)
            else:
                month=np.random.randint(1,12+1)
            if year == 3 and month == 3:
                day=np.random.randint(5,31+1)
            elif month == 2:
                day=np.random.randint(1,28+1)
            elif month == 4 or month == 6 or month == 9 or month == 11:
                day=np.random.randint(1,30+1)
            else:
                day=np.random.randint(1,31+1)
            
            
            StartPoint = Data.index.get_loc('201'+str(year)+'-'+str(month)+'-'+str(day)+' 00:00',method='nearest')
            EndPoint = Data['201'+str(year)+'-'+str(month)+'-'+str(day)].shape[0]
            fig, ax = plt.subplots()  
            ax.plot(Data.index[StartPoint:StartPoint+EndPoint], Data.Diesel['201'+str(year)+'-'+str(month)+'-'+str(day)])
            # format the ticks
            ax.xaxis.set_major_locator(hoursM)
            ax.xaxis.set_major_formatter(hoursFmt)
            ax.xaxis.set_minor_locator(hours)
            # round to nearest years...
            datemin = np.datetime64(Data.index[StartPoint], 'm')
            datemax = np.datetime64(Data.index[StartPoint+EndPoint-1], 'm')
            # + np.timedelta64(1, 'D')
            ax.set_xlim(datemin, datemax)
            ax.grid(which='major', alpha = 1, color='black')
            ax.grid(which='minor', alpha = 0.05, color ='black')
            fig.autofmt_xdate()
            Diesel = Data.Diesel['201'+str(year)+'-'+str(month)+' - '+str(day)]
            Water = Data.Water['201'+str(year)+'-'+str(month)+' - '+str(day)]+Diesel
            Wind = Data.Wind['201'+str(year)+'-'+str(month)+' - '+str(day)]+Water
            #plt.plot(Diesel,color=Color1)
            plt.plot(Water,color=Color2)
            plt.plot(Wind,color=Color3)
            plt.fill_between(Wind.index,Wind, Water, facecolor=Color3, alpha=1)
            plt.fill_between(Water.index,Water, Diesel, facecolor=Color2, alpha=1)
            plt.fill_between(Diesel.index,Diesel, 0, facecolor=Color1, alpha=1)
            plt.title('201'+str(year)+' - '+monthDict[month]+ ' - '+str(day))
            plt.ylim([-1,105])
            plt.xlabel('Time [date]')
            plt.ylabel('Total production [%]')
            plt.legend(['Diesel','Water','Wind'])
    
            plt.savefig(GraphPath+'Daily/'+'Share201'+str(year)+'-'+str(month)+'-'+str(day)+'.png')
            plt.close(fig)
            progbar.update(pcount)
        #pbar.finish()
    progbar = ProgBar.Progbar(target = toCreate, newline_on_end = False,
                         text_description='Done: ')
    if toCreate > 0:
        progbar.update(pcount)
        
        
def ScatterMatrix(GraphPath, Data, Name):
    """Creates scatter matrix
    
    Creates a scatter matrix of the data
    
    Parameters
    ----------
    GraphPath : str
        Path to save the graphs.
    
    Data : array-like
        The data to graph.
    
    Name : str
        The name of the saved graph
    """    
    pd.plotting.scatter_matrix(Data, alpha=0.2, figsize=(Data.shape[1]+1, Data.shape[1]+1))
    plt.savefig(GraphPath+Name+'.png')
    
    
def WeatherGraphs(GraphPath, Weather_Data):
    """Creates monthly weather graphs
    
    Creates monthly graphs showing the weather patterns
    
    Parameters
    ----------
    GraphPath : str
        Path to save the graphs.
    
    Weather_Data : array-like
        The weather data to graph.
    """  
    majorTick = mdates.DayLocator(interval=7)  # every month
    majorFmt = mdates.DateFormatter('%d/%m')
    minorTick = mdates.DayLocator() # every day
    if not os.path.isdir(GraphPath+'Monthly/'):
        print('Monthly folder not found, creating it')
        os.makedirs(GraphPath+'Monthly')
    if not os.path.isfile(GraphPath+'Monthly/Weather2018-3.png'):
        widgets = ['Creating monthly weather graph: ', progressbar.Percentage(),
                   ' ',progressbar.Bar(marker='∎',left='[',right=']'),
                   ' ', progressbar.AdaptiveETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=60)
        pbar.start()
        month=3
        year=3
        count = 0
        done = False
        #print('Creating Monthly weather graph [%]')
        while not done:
            pbar.update(count)
            count = count+1
            #if count % 6 == 0:
            #    print('\u220E', end='', flush=True)
            StartPoint = Weather_Data.index.get_loc('201'+str(year)+'-'+str(month)+'-'+'01 00:00',method='nearest')
            EndPoint = Weather_Data['201'+str(year)+'-'+str(month)].shape[0]
            fig, ax = plt.subplots()  
            ax.plot(Weather_Data.index[StartPoint:StartPoint+EndPoint], Weather_Data.W_speed_10['201'+str(year)+'-'+str(month)],color=Color2)
            # format the ticks
            ax.xaxis.set_major_locator(majorTick)
            ax.xaxis.set_major_formatter(majorFmt)
            ax.xaxis.set_minor_locator(minorTick)
            datemin = np.datetime64(Weather_Data.index[StartPoint], 'm')# round to nearest years...
            datemax = np.datetime64(Weather_Data.index[StartPoint+EndPoint-1], 'm')
            ax.set_xlim(datemin, datemax)
            ax.grid(which='major', alpha = 1, color='black')
            ax.grid(which='minor', alpha = 0.05, color ='black')
            fig.autofmt_xdate()
            plt.title('201'+str(year)+' - '+monthDict[month])
            plt.ylim([0,110])
            plt.xlabel('Time [date]')
            plt.ylabel('Wind Speed [km/h]')
            plt.savefig(GraphPath+'Monthly/'+'Weather201'+str(year)+'-'+str(month)+'.png')
            plt.close(fig)
            if month == 12:
                month = 1
                year = year + 1
            else:
                month = month+1
            if year == 8 and month == 4:
                done = True
                #print('')
                pbar.finish()
                
def PlotHistory(predictions, ground_truth, GraphPath,plotDict={0:'Diesel',1:'Water',2:'Wind',3:'Total'}):
    """Plot history
    
    Plot the history and ground truth after predictions
    
    Parameters
    ----------
    predictions : array-like
        The predictions from the model.
    
    ground_truth : array-like
        The true values.
        
    GraphPath : str
        Path to save the graphs
        
    plotDict : dict , optional
        Description of the data for loop.
        (default = {0:'Diesel',1:'Water',2:'Wind',3:'Total'})
    """  
    for n in range(predictions.shape[1]):
        fig, ax = plt.subplots()  
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.05, color ='black')
        plt.plot(ground_truth.index,ground_truth[plotDict[n]],'^',color=Color2)
        plt.plot(ground_truth.index,predictions[:,n], '*',color=Color1)
        
        plt.fill_between(ground_truth.index,ground_truth[plotDict[n]], predictions[:,n], facecolor='red', alpha=1, step='post')
        plt.title(plotDict[n])
        plt.xlabel('Time [date]')
        plt.ylabel('Production [%]')
        plt.legend([plotDict[n]+'_true',plotDict[n]+'_pred'])
        fig.set_size_inches(16, 9, forward = True)
        plt.savefig(GraphPath+plotDict[n]+'.png')
        plt.close(fig)
    
        
def PlotHistory2018(predictions, ground_truth, GraphPath,plotDict={0:'Diesel',1:'Water',2:'Wind',3:'Total'}):
    """Plot history 2018
    
    Plot the history and ground truth after predictions of 2018
    
    Parameters
    ----------
    predictions : array-like
        The predictions from the model.
    
    ground_truth : array-like
        The true values.
        
    GraphPath : str
        Path to save the graphs
        
    plotDict : dict , optional
        Description of the data for loop.
        (default = {0:'Diesel',1:'Water',2:'Wind',3:'Total'})
    """  
    majorTick = mdates.DayLocator(interval=7)  # every month
    majorFmt = mdates.DateFormatter('%d/%m')
    minorTick = mdates.DayLocator() # every day
    predictions = predictions[-len(ground_truth['2018-03']):]
    ground_truth = ground_truth['2018-03']
    for n in range(predictions.shape[1]):
        fig, ax = plt.subplots()  
        ax.xaxis.set_major_locator(majorTick)
        ax.xaxis.set_major_formatter(majorFmt)
        ax.xaxis.set_minor_locator(minorTick)
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.5, color ='black')
        plt.plot(ground_truth.index,ground_truth[plotDict[n]],'^',color=Color2)
        plt.plot(ground_truth.index,predictions[:,n], '*',color=Color1)
        
        plt.fill_between(ground_truth.index,ground_truth[plotDict[n]], predictions[:,n], facecolor='red', alpha=1, step='post')
        plt.title(plotDict[n])
        plt.xlabel('Time [date]')
        plt.ylabel('Production [%]')
        plt.legend([plotDict[n]+'_true',plotDict[n]+'_pred'])
        fig.set_size_inches(16, 9, forward = True)
        plt.savefig(GraphPath+plotDict[n]+'2018.png')
        plt.close(fig)
    

def PlotHistoryDiff(predictions, ground_truth, GraphPath,plotDict={0:'Diesel',1:'Water',2:'Wind',3:'Total'}):
    """Plot history 2018
    
    Plot the history and ground truth after predictions of 2018
    
    Parameters
    ----------
    predictions : array-like
        The predictions from the model.
    
    ground_truth : array-like
        The true values.
        
    GraphPath : str
        Path to save the graphs
        
    plotDict : dict , optional
        Description of the data for loop.
        (default = {0:'Diesel',1:'Water',2:'Wind',3:'Total'})
    """  
    majorTick = mdates.DayLocator(interval=7)  # every month
    majorFmt = mdates.DateFormatter('%d/%m')
    minorTick = mdates.DayLocator() # every day
    predictions = predictions[-len(ground_truth['2018-03']):]
    ground_truth = ground_truth['2018-03']
    for n in range(predictions.shape[1]):
        fig, ax = plt.subplots()  
        ax.xaxis.set_major_locator(majorTick)
        ax.xaxis.set_major_formatter(majorFmt)
        ax.xaxis.set_minor_locator(minorTick)
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.5, color ='black')
        plt.stem(ground_truth.index,predictions[:,n]-ground_truth[plotDict[n]])        
        plt.title(plotDict[n]+' difference')
        plt.xlabel('Time [date]')
        plt.ylabel('Production [MW]')
        fig.set_size_inches(16, 9, forward = True)
        plt.savefig(GraphPath+plotDict[n]+'2018_diff.png')
        plt.close(fig) 
                   
        
def PlotHistory2018percent(predictions, ground_truth, GraphPath,plotDict={0:'Diesel',1:'Water',2:'Wind',3:'Total'}):
    """Plot history 2018
    
    Plot the history and ground truth after predictions of 2018
    
    Parameters
    ----------
    predictions : array-like
        The predictions from the model.
    
    ground_truth : array-like
        The true values.
        
    GraphPath : str
        Path to save the graphs
        
    plotDict : dict , optional
        Description of the data for loop.
        (default = {0:'Diesel_per',1:'Water_per',2:'Wind_per',3:'Total_per'})
    """  
    majorTick = mdates.DayLocator(interval=7)  # every month
    majorFmt = mdates.DateFormatter('%d/%m')
    minorTick = mdates.DayLocator() # every day
    predictions = predictions[-len(ground_truth['2018-03']):]
    ground_truth = ground_truth['2018-03']
    for n in range(predictions.shape[1]):
        fig, ax = plt.subplots()  
        ax.xaxis.set_major_locator(majorTick)
        ax.xaxis.set_major_formatter(majorFmt)
        ax.xaxis.set_minor_locator(minorTick)
        ax.grid(which='major', alpha = 1, color='black')
        ax.grid(which='minor', alpha = 0.5, color ='black')
        plt.plot(ground_truth.index,ground_truth[plotDict[n]],'^',color=Color2)
        plt.plot(ground_truth.index,predictions[:,n], '*',color=Color1)
        
        plt.fill_between(ground_truth.index,ground_truth[plotDict[n]], predictions[:,n], facecolor='red', alpha=1, step='post')
        plt.title(plotDict[n])
        plt.xlabel('Time [date]')
        plt.ylabel('Production [%]')
        plt.legend([plotDict[n]+'_true',plotDict[n]+'_pred'])
        fig.set_size_inches(16, 9, forward = True)
        plt.savefig(GraphPath+plotDict[n]+'_per2018.png')
        plt.close(fig)
    
        
def CVGraph(ls, test, start, GraphPath,rangevar=None,name=None,maxtot=None,log=False):
    """Create graphs from CV data
    
    Creates a figure with 2 subplots.
    Subplot 1 contains the Val MAE data (min,mid,max)
    Subplot 2 contains the epoch data (min, mid, max)
    
    Parameters
    ----------
    ls : dict
        listOfErros dictionary from CrossValidation.
    
    test : str
        The type of test run.
    
    start : int
        Start argument from CV, to fix numbering
    
    GraphPath : str
        Path to save the graphs    
        
    Returns
    -------
    results : dict
        Dictionary of results from ls
        
    fig : figure
        Matlpotlib figure object of graphs
    """  
    if name is None:
        name = test
    else:
        pass
    if rangevar is None:
        rangevar = range(len(ls['runs']))
    else:
        #rangevar = rangevar
        pass
    if maxtot is not None:
        pass

    minres = dict()
    maxres = dict()
    midres = dict()
    res = {'min': minres, 'max': maxres, 'mid': midres}
    for y in rangevar:
        y = y+start
        if test == 'nodes':
            y = y*256
        num=0
        minval = 9999999999
        maxval = 0
        for x in range(len(ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.').rstrip('.')])):
            num += ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['val_MAE']/maxtot*100
            if ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['val_MAE']/maxtot*100 <= minval:
                minval = ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['val_MAE']/maxtot*100
            if ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['val_MAE']/maxtot*100 >= maxval:
                maxval = ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['val_MAE']/maxtot*100
            res['mid'][y] = (num/len(ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]))
            res['min'][y] = minval
            res['max'][y] = maxval
        
    minres = dict()
    maxres = dict()
    midres = dict()
    res2= {'min': minres, 'max': maxres, 'mid': midres}
    for y in rangevar:
        y = y+start
        if test == 'nodes':
            y = y*256
        num=0
        minval = 9999999999
        maxval = 0
        for x in range(len(ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')])):
            num += ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['Epoch']
            if ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['Epoch'] <= minval:
                minval = ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['Epoch']
            if ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['Epoch'] >= maxval:
                maxval = ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]['run'+str(x+1)]['Epoch']
            res2['mid'][y] = (num/len(ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]))
            res2['min'][y] = minval
            res2['max'][y] = maxval
            

    fig, (ax, ax2) = plt.subplots(2, sharex=True, frameon=False)  
    if test == 'layers':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    if test == 'nodes':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(256))
    ax.plot(*zip(*sorted(res['max'].items())),label='max')        
    ax.plot(*zip(*sorted(res['mid'].items())),label='mid')
    ax.plot(*zip(*sorted(res['min'].items())),label='min')
    #ax.set_xscale('log')
    ax.grid(which='major', alpha = 0.25, color='black')
    ax.grid(which='minor', alpha = 0.1, color='black')
    #ax.set_xlabel(test.capitalize())
    ax.set_ylabel('Validation score\n [% of maximum total production]')
    ax.legend()
    
    #for j in range(len(res['mid'])):
    #    ax.annotate('{:.5f}'.format(res['mid'][j+start]),xy=(j+start,res['mid'][j+start]))
        
    
       
    #ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax2.plot(*zip(*sorted(res2['max'].items())),label='max')        
    ax2.plot(*zip(*sorted(res2['mid'].items())),label='mid')
    ax2.plot(*zip(*sorted(res2['min'].items())),label='min')
    
    if log:
        ax2.set_xscale('log')
    ax2.grid(which='major', alpha = 0.25, color='black')
    
    ax2.set_xlabel(test.capitalize())
    ax2.set_ylabel('Epochs')
    
    #ax2.legend()
    fig.subplots_adjust(hspace=0.1)
    ax.set_title(str(len(ls['runs'][test+format(y,'.10f').rstrip('0').rstrip('.')]))
                +'-fold Cross Validation of '+name)
    
    plt.savefig(GraphPath+name+'_CV.png')
    results = {'Val': res, 'Epoch': res2}
    return results, fig

def closeFigs():
    """Close all figures
    """
    plt.close('all')
    
def openFigs():
    """Opens all figures
    """
    plt.show()
    
def plotFromReadme(data, plots):
    """Plot graphs from readme data
    
    Takes the data from the readReadme type functions, and plots a graph
    
    Parameters
    ----------
    data : dict
        The history dictionary from readReadme functions.
    
    plots : array-like
        The keys from the history dictionary.
        
    Returns
    -------
    fig : figure object
        A matplotlib figure object
    """
    plots2=['Train','Validation']
    fig, ax = plt.subplots(1)
    for n in range(len(plots)):
        ax.plot(data['Epoch'],data[plots[n]],label=plots2[n])        
    ax.grid(which='major', alpha = 0.25, color='black')
        
    #ax.set_xlabel(test.capitalize())
    ax.set_ylabel('Loss\n[% maximum total production ')
    ax.set_xlabel('Epoch')
    ax.legend()
    
    fig.suptitle('Final model evaluation')
    fig.set_size_inches(16, 9, forward = True)
    plt.savefig(os.getcwd().replace('\\','/')+'final2'+'.png')
    return fig

def PlotScatter(ydata, xdata, path, row, columns, ylabel, xlabels, xlim, title):
    
    fig, ax = plt.subplots(nrows=row, ncols=columns, sharey=True, frameon=False)
    
    for x in range(0,row):
        for y in range(0,columns):
            if row == 1:
                if columns==1:
                    ax.scatter(xdata[y], ydata)
                    ax.set_xlabel(xlabels[y])
                    ax.set_xlim(xlim[x])
                else:
                    ax[y].scatter(xdata[y], ydata)
                    ax[y].set_xlabel(xlabels[y])
                    ax[y].set_xlim(xlim[x])
            else:
                ax[x][y].scatter(xdata[x][y], ydata)
                ax[x][y].set_xlabel(xlabels[x][y])
                ax[x][y].set_xlim(xlim[x])
        if row == 1:
            if columns == 1:
                ax.set_ylabel(ylabel)
            else:
                ax[0].set_ylabel(ylabel)
        else:
            ax[x][0].set_ylabel(ylabel)
    
    fig.suptitle(title)
    
    fig.set_size_inches(16, 9, forward = True)
    plt.savefig(path+'Scatter'+'.png')
    plt.close(fig)
    return fig

def Graph(Data,GraphPath,name=''):
    years = mdates.MonthLocator()   # every year
    months = mdates.DayLocator()  # every month
    yearsFmt = mdates.DateFormatter('%D')
    plt.rcParams["figure.figsize"] = [16,9]
    
    #print('Creating diesel overview graph')
    fig, ax = plt.subplots()    
    ax.plot(Data.index, Data.Wind,color=Color1)
    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    # round to nearest years...
    datemin = np.datetime64(Data.index[0], 'D')
    datemax = np.datetime64(Data.index[-1], 'D') + np.timedelta64(1, 'D')
    ax.set_xlim(datemin, datemax)
    plt.title('Timeseries')
    plt.xlabel('Time [date]')
    plt.ylabel('Wind production [% of cleaned total]')
    ax.grid(which='major', alpha = 1, color='black')
    ax.grid(which='minor', alpha = 0.05, color ='black')
    fig.autofmt_xdate()
    plt.savefig(GraphPath+name+'.png')