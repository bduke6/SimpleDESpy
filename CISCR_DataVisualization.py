# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:50:28 2021

@author: michael.a.yereniuk
"""
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
import random
import configparser

import os.path

CONFIG_FILE_NAME = "../../../../Downloads/CONFIG.csv"
VISUALIZATION_REFRESHRATE = 100
RUN_NO=0

def getDataByID(aDF, anID, idLabel = 'AgentID'):
    '''
    Return dataframe with only values of a specific id
    '''
    tmpDF = aDF[aDF[idLabel]==anID]
    return tmpDF

def plotRegion(vertices, centerOfMass, color="Blue", alpha=0.4):
        '''
        Plot region and center of mass
        '''
        plt.scatter(centerOfMass[0], centerOfMass[1],color='Purple', s=30)
        ax = plt.gca()
        poly = plt.Polygon(vertices, color=color, alpha=alpha)
        ax.add_patch(poly)
        
        
def loadData(filePathStr="", run=0):
    '''
    Load dataframes and return
    '''
    agentTDFName = filePathStr+"agentT_run"+str(run)+".json"
    if isFileExists(agentTDFName):
        agentTDF = pd.read_json(agentTDFName)
    else:
        agentTDF = None
    agentODFName = filePathStr+"agentO_run"+str(run)+".json"
    if isFileExists(agentODFName):
        agentODF = pd.read_json(agentODFName)
    else:
        agentODF = None
    sensorDFName = filePathStr+"sensor_run"+str(run)+".json"
    if isFileExists(sensorDFName):
        sensorDF = pd.read_json(sensorDFName)
    else:
        sensorDF = None
    regionDFName = filePathStr+"region_run"+str(run)+".json"
    if isFileExists(regionDFName):
        regionDF = pd.read_json(regionDFName)
    else:
        regionDF = None
    guessDFName = filePathStr+"guess_run"+str(run)+".json"
    if isFileExists(guessDFName):
        guessDF = pd.read_json(guessDFName)
    else:
        guessDF = None
    return agentTDF, agentODF, sensorDF, regionDF, guessDF

def isFileExists(fileStr):
    '''
    Check if file exists
    '''
    return os.path.isfile(fileStr)

# Get domain and data parameters from CONFIG.ini
#Read CONFIG.ini file
config_obj = configparser.ConfigParser()
config_obj.read(CONFIG_FILE_NAME)
runData = config_obj['RunInfo_Parameters']
domainData = config_obj['Domain_Parameters']

# Domain Parameters
xMin = float(domainData['xMin'])
xMax = float(domainData['xMax'])
yMin = float(domainData['yMin'])
yMax = float(domainData['yMax'])
# Data load parameters
filePath = runData['filePath']
numRuns = int(runData['numRuns'])

run = RUN_NO

# Load data files
agentTDF, agentODF, sensorDF, regionDF, guessDF = loadData(filePathStr=filePath, run=run)
agentTDF = agentTDF.sort_values(by=['Time'])

# Plot regions
fig = plt.figure(2)
fig.set_dpi(100)
fig.set_size_inches(8, 8)
ax = plt.gca()
ax.cla()
for idx in range(len(regionDF)):
    color=(random.uniform(0.1,1),random.uniform(0.1,1),random.uniform(0.1,1))
    plotRegion(regionDF['Vertices'][idx], regionDF['CenterOfMass'][idx], color=color)

# Initialize axis and title
dx, dy = xMax-xMin, yMax-yMin
ax.set_xlim([xMin-0.1*dx, xMax+0.1*dx])
ax.set_ylim([yMin-0.1*dy, yMax+0.1*dy])
ax.set_title("t=0")     
#plt.plot(x,y)

# Visualization patch data
patches_ac = []
patches_rings = []
patches_opaque = []
patches_region = []
deadID = []

# Initialize visualization patches
init_AgentT = agentTDF[agentTDF['Time']==0.0]
for idx in range(len(init_AgentT)):
    aLoc = init_AgentT['Location'][idx]
    center = (aLoc[0], aLoc[1])
    radius = init_AgentT['SensorRadius'][idx]
    FOR = init_AgentT['FOR'][idx]
    direction = init_AgentT['SensorAngle'][idx]
    leftAngle = int(direction-FOR/2)
    rightAngle = int(direction+FOR/2)
    agent_clone = patches.Circle(center, 0.75, fc='b')
    agent_ring = patches.Wedge(center, radius,leftAngle,rightAngle,fill=False)#plt.Circle(center, radius, fill=False)
    patches_ac.append(agent_clone)
    patches_rings.append(agent_ring)
    ax.add_patch(agent_clone)
    ax.add_patch(agent_ring)
init_AgentO = agentODF[agentODF['Time']==0.0]
hashID = {}
for idx in range(len(init_AgentO)):
    aLoc = init_AgentO['Location'][idx]
    center = (aLoc[0], aLoc[1])
    agent_o = patches.Circle(center,0.1, fc='r')
    patches_opaque.append(agent_o)
    ax.add_patch(agent_o)
    hashID[init_AgentO['AgentID'][idx]]=idx
for idx in range(len(regionDF)):
    vertices = regionDF["Vertices"][idx]
    color=(random.uniform(0.1,1),random.uniform(0.1,1),random.uniform(0.1,1))
    poly = plt.Polygon(vertices, color=color, alpha=0.1)
    patches_region.append(poly)
    ax.add_patch(poly)
def init():
    return patches_ac+patches_rings+patches_opaque+patches_region

def animationManage(time):
    '''
    Update the animation, based on the time and log file data.
    '''
    time_AgentT = agentTDF[agentTDF['Time']==float(time)]
    time_AgentO = agentODF[agentODF['Time']==float(time)]
    time_Sensor = sensorDF[sensorDF['Time']==float(time)]
    agtID = time_AgentT['AgentID'].unique()
    # Plot transparency agents
    for idx in range(len(time_AgentT)):
        myAgent = time_AgentT[time_AgentT['AgentID']==agtID[idx]]
        aLoc = myAgent.iloc[0].Location
        # Determine sensor parameters
        direction = myAgent.iloc[0].SensorAngle
        FOR = myAgent.iloc[0].FOR
        leftAngle = int(direction-FOR/2)
        rightAngle = int(direction+FOR/2)
        center = [aLoc[0], aLoc[1]]
        patches_ac[idx].set_center(center) #= center
        patches_rings[idx].set_center(center) # = center
        patches_rings[idx].set_theta1(leftAngle)
        patches_rings[idx].set_theta2(rightAngle)
        patches_rings[idx]._recompute_path()
    # Plot opaque agents
    for idNumber in time_AgentO['AgentID'].unique():
        myData = getDataByID(time_AgentO, idNumber)
        # Determine if agent is inactive
        if False in myData['Active'].unique():
            idx = hashID[idNumber]
            patches_opaque[idx].set_color("Black")
            deadID.append(idx)
            # Make entire region dark
            regionID = myData['RegionID'].unique()
            patches_region[regionID[0]].set_color("Black")
            patches_region[regionID[0]].set_alpha(0.5)            
    for idNumber in agentODF['AgentID'].unique():
        # Check any agents in time_Sensor
        idx = hashID[idNumber]
        if not idx in deadID:
            # Determine whether agent was acquired by at least one sensor
            if idNumber in time_Sensor['TgtID'].unique():
                myAgent = getDataByID(time_Sensor, idNumber, 'TgtID')
                acq = 0
                for i in range(len(myAgent)):
                    acq+=myAgent.iloc[i].Acquired
                if acq>=1:
                    patches_opaque[idx].set_color("Blue")
                if acq==0:
                    patches_opaque[idx].set_color("Green")
            else:
                patches_opaque[idx].set_color("Red")
    # Update time in plot title
    ax.set_title(f"t = {time:5.2f}")
    return patches_ac+patches_rings+patches_opaque+patches_region#[]+[]

# Run animation
anim = animation.FuncAnimation(fig, animationManage,
                               init_func=init,
                               frames=agentTDF['Time'].unique(),
                               interval=VISUALIZATION_REFRESHRATE,
                               blit=False,
                               repeat=False)
fig.canvas.draw()
plt.show()