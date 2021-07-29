# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:53:17 2021

@author: michael.a.yereniuk
"""
import helper_functions as hf
from math import pi, sqrt, floor, cos, sin
from scipy.spatial import Voronoi
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import configparser

import os.path
import logging

CONFIG_FILE_NAME = "../../../../Downloads/CONFIG2.csv"

# Initialize seed for program testing
#random.seed(1)
#np.random.seed(10)

class Agent():
    '''
    Class for transparent and opaque agents.  Transparency agents try to find 
    center of region by observing opaque objects.
    '''
    def __init__(self, _location, _searchAngle, _FOR, _acquisitionDistance,
                 _idNumber, _acquisitionProbability=1.0):
        self.idNumber = _idNumber
        self.location = _location
        self.direction = None
        self.searchAngle = _searchAngle
        self.FOR = _FOR
        self.acquisitionDistance = _acquisitionDistance
        self.acquisitionProbability = _acquisitionProbability
    
    def initLocation(self, boundingBox):
        '''
        Initialize location, searchAngle, direction
        Bounding Box is of form [xMin, xMax, yMin, yMax]
        '''
        xMin,xMax,yMin,yMax = boundingBox
        center = (0.5*(xMin+xMax), 0.5*(yMin+yMax))
        angle = np.random.uniform(0,2*pi,1)
        radius = 1.2*sqrt(hf.getDistanceSqrd(center, (xMin,yMin)))
        self.location = [center[0]+radius*cos(angle), center[1]+radius*sin(angle)]
        # TODO: choose direction based on bounding box
        if xMin<self.location[0]<xMax:
            if self.location[0]<yMin:
                self.direction = pi/2
            else:
                self.direction = -pi/2
        elif yMin<self.location[1]<yMax:
            if self.location[0]<xMin:
                self.direction = 0
            else:
                self.direction = pi
        else:
            self.direction = hf.getAngle(self.location, center)
        
    def updateAgent(self, agentList):
        '''
        Update agent for next time-step.
        '''
        # Move agent/change search Angle
        self.updateAgentLocation()
        # Target acquisition
        self.updateAgentInfo(agentList)
    
    def updateAgentAcquisition(self, agentList):
        '''
        Update agent's sensor acquisition data.
        '''
        self.acqTarget, self.notAcqTarget = self.getAgentsFromSensor(agentList)
    
    def updateSpeedDirection(self, speed, direction, error=0.0):
        '''
        Update an agent's speed and direction.  Note, direction should
        be an angle in radians and will be converted to unit vector tuple (x,y).
        '''
        self.speed = speed
        self.direction = direction
        self.moveError = error
            
    def updateAgentLocation(self, dt):
        '''
        Update agent location by moving a distance after time dt along current
        direction.
        '''
        self.direction += np.random.uniform(-self.moveError/2,self.moveError/2,[1])
        self.searchAngle = self.direction
        self.location[0]+=dt*self.speed*cos(self.direction)
        self.location[1]+=dt*self.speed*sin(self.direction)
    
    def updateAgentSearchAngle(self, aAngle):
        '''
        Update agent's search angle
        '''
        self.searchAngle = aAngle
        
    def getEstimatedLocation(self):
        '''
        Use target locations to update 'guess' about center of mass
        '''
        # TODO: Figure out how to get knowledge. Is average location
        # sufficient, or should we have a Kalman filter use memory?
        if len(self.acqTarget)>0:
            acqTargetLocations = []
            for aTarget in self.acqTarget:
                acqTargetLocations.append(aTarget.location)
            return hf.locAverage(np.array(acqTargetLocations))
        else:
            return None
    
    def getAgentsFromSensor(self, agentOList):
        '''
        Get opaque agents based on sensor acquisition.  Can only acquire
        agents if region is active (not "dark" from previous actions).
        '''
        acqAgents = []
        notAcqAgents = []
        for aRegion in agentOList:
            if aRegion.isActive:
                # Only sense if region is active
                for anAgent in agentOList[aRegion]:
                    checkAcq = self.isSensorAcquire(anAgent.location)
                    if checkAcq == 1:
                        # If sensor acquires anAgent, add agent to list
                        acqAgents.append(anAgent)
                    elif checkAcq == 0:
                        # If in sensor area, but not acquire, add agent to list
                        notAcqAgents.append(anAgent)
        return acqAgents, notAcqAgents
    
    def isSensorAcquire(self, aLoc):
        '''
        Determine whether a sensor can acquire a location
        Can change based on model
        '''
        # Check if distance is small enough
        dist = hf.getDistanceSqrd(self.location, aLoc)
        if dist<self.acquisitionDistance**2:
            # Check if angle is small enough
            angle = hf.getAngle(self.location, aLoc)
            if abs(hf.angleDifference(angle,self.searchAngle)) < self.FOR/2:
                # Check if acquisition probability
                probAcq = np.random.uniform(0.0, 1.0, 1)
                if probAcq<self.acquisitionProbability:
                    return 1
                else:
                    return 0
        return -1
    
    def plotAgent(self, color="Black", size=10):
        '''
        Plot agent in window
        '''
        plt.scatter(self.location[0],self.location[1],color=color, s=size)
        
    def plotSensor(self, color="Black", alpha=0.2):
        '''
        Plot sensor region
        '''
        ax = plt.gca()
        center = (self.location[0], self.location[1])
        leftAngle = (self.searchAngle-self.FOR/2)*180.0/pi
        rightAngle = (self.searchAngle+self.FOR/2)*180.0/pi
        wedge = Wedge(center, self.acquisitionDistance, leftAngle, rightAngle, 
                      color=color,alpha=alpha)
        ax.add_patch(wedge)

class Region():
    ''' 
    Class with information about a particular region (polygonal subset
    of bounding box)
    '''
    def __init__(self, _vertices, _idNumber):
        self.idNumber = _idNumber
        self.vertices = _vertices
        # Compute geometric values
        self.area = abs(hf.areaPolygon(self.vertices))
        self.centerOfMass = hf.centerOfMassPolygon(self.vertices)
        self.bounds = None
        # If region has active opaque agents (True) or not (False)
        self.isActive = True
        
    def plotRegion(self, color="Blue", alpha=0.4):
        '''
        Plot region and center of mass
        '''
        ax = plt.gca()
        poly = plt.Polygon(self.vertices, color=color, alpha=alpha)
        ax.add_patch(poly)
        plt.scatter(self.centerOfMass[0], self.centerOfMass[1],color='Red', s=30)
        
    def getRandomPoints(self, numPoints=1, distribution="Uniform", param=1.0):
        '''
        Generate random point inside region
        '''
        pointList = []
        if self.bounds==None:
            self.updateBounds()
        # Create random point
        if distribution=="Uniform":
            x = np.round(np.random.uniform(self.bounds[0], self.bounds[1], numPoints), 2)
            y = np.round(np.random.uniform(self.bounds[2], self.bounds[3], numPoints), 2)
        if distribution=="Normal":
            x = np.round(np.random.normal(self.centerOfMass[0], param, numPoints), 2)
            y = np.round(np.random.normal(self.centerOfMass[1], param, numPoints), 2)
        pointList = np.transpose(np.array([x,y]))
        for idx in range(numPoints):
            x,y = pointList[idx,:]
            # If point outside region, get another point
            while not hf.isPointInPolygon(self.vertices,x,y):
                if distribution=="Uniform":
                    x = np.round(np.random.uniform(self.bounds[0], self.bounds[1], 1), 2)
                    y = np.round(np.random.uniform(self.bounds[2], self.bounds[3], 1), 2)
                elif distribution=="Normal":
                    x = np.round(np.random.normal(self.centerOfMass[0], param, 1), 2)
                    y = np.round(np.random.normal(self.centerOfMass[1], param, 1), 2)
            # Append point to list
            pointList[idx,0]=x
            pointList[idx,1]=y
        # Return points
        return pointList
    
    def updateBounds(self):
        '''
        Update the min and max values of the x and y coordinates of the region
        '''
        self.bounds = [0,0,0,0]
        x = [aPoint[0] for aPoint in self.vertices]
        y = [aPoint[1] for aPoint in self.vertices]
        self.bounds[0] = min(x)
        self.bounds[1] = max(x)
        self.bounds[2] = min(y)
        self.bounds[3] = max(y)
        
    
        
class PolygonPartition():
    ''' 
    Class that partitions a bounding box into numPartition regions 
    '''
    def __init__(self, _numPartitions, minX=0, maxX=1, minY=0, maxY=1, minAreaBound=0.0):
        self.numPartitions = _numPartitions
        self.boundingBox = np.array([minX, maxX, minY, maxY])
        self.regions = self.getRegions(minAreaBound = minAreaBound)
        
    def getPoints(self, minAreaBound = 0):
        '''
        Obtain points in domain
        '''
        xMin, xMax, yMin, yMax = self.boundingBox
        if minAreaBound == 0:
            return hf.create_random_points(self.numPartitions, xMin, xMax, yMin, yMax)
        else:
            sideLen = sqrt(minAreaBound)*4
            # Ensure grid lengths are >= sideLen
            nx = floor((xMax-xMin)/sideLen)
            ny = floor((yMax-yMin)/sideLen)
            if nx*ny<self.numPartitions:
                return None
            idxList = [(idxX,idxY) for idxX in range(nx) for idxY in range(ny)]
            grids = random.sample(idxList, self.numPartitions)
            dx = (xMax-xMin)/nx
            dy = (yMax-yMin)/ny
            myPoints = np.zeros((self.numPartitions,2))
            idx = 0
            for aSample in grids:
                x = np.round(np.random.uniform(xMin+aSample[0]*dx, xMin+(aSample[0]+1)*dx, 1), 2)
                y = np.round(np.random.uniform(yMin+aSample[1]*dy, yMin+(aSample[1]+1)*dy, 1), 2)
                myPoints[idx,0] = x
                myPoints[idx,1] = y
                idx+=1
            return np.array(myPoints)
        
    def getRegions(self, minAreaBound = 0.0):
        '''
        Obtain list of regions
        '''
        # Get Points to generate regions
        myPoints = self.getPoints(minAreaBound=minAreaBound)
        if myPoints is None:
            print("Too many points to ensure minimum area.")
            return None
        myPoints = hf.reflectPoints(myPoints, self.boundingBox)
        # Create voronoi partition of region
        vor = Voronoi(myPoints)
        # Generate Region objects
        boundRegions = hf.filterRegions(vor, self.boundingBox, eps=0.1)
        myRegions = []
        idx = 0
        for aRegion in boundRegions:
            myVertices = vor.vertices[aRegion]
            myRegions.append(Region(myVertices,idx))
            idx += 1
        return myRegions
    
    def logRegions(self):
        '''
        Save log of regions and metrics
        '''
        for aRegion in self.regions:
            logging.info("ID %s, CenterOfMass (%s,%s), Area %s",str(aRegion.idNumber),
                         str(aRegion.centerOfMass[0]),str(aRegion.centerOfMass[1]), 
                         str(aRegion.area))
            #print(aRegion.idNumber, aRegion.centerOfMass, aRegion.area)
        
    
class GlobalInfo():
    '''
    Class that holds global information that can be passed to any agent or
    method.
    '''
    def __init__(self, guessTolerance):
        # Regions
        self.regions = []
        # Opaque agents stored with region {regionObject: set of agentObjects}
        self.agent_O = {}
        # Transparency agent
        self.agent_T = None
        # Estimated location data stored with region {regionObject: estimated location}
        self.estimatedLocations = {}
        self.guessTolerance = guessTolerance
        # Agent ID counter
        self.agentIDCount = 0
        # Current time
        self.currentTime = 0.0
        # Data Frames
        self.initializeDataFrames()
        
    def initializeDataFrames(self):
        '''
        Initialize data frame columns
        '''
        agentTCols = ["AgentID", "Time", "Location", "FOR", "SensorAngle",
                      "SensorRadius", "Speed", "Direction"]
        agentOCols = ["AgentID", "Time", "Location", "RegionID", "Active"]
        sensorCols = ["AgentID", "Time", "TgtID", "TgtLocation"]
        regionCols = ['RegionID', 'Vertices', 'Area', 'CenterOfMass',
                      'BestGuessLocation', 'BestGuessError']
        guessCols = ["AgentID", "Time", "Location", "MinDistance", 
                     "MinRegionID", "GuessIDs"]
        self.agentTDF = pd.DataFrame(columns = agentTCols)
        self.agentODF = pd.DataFrame(columns = agentOCols)
        self.sensorDF = pd.DataFrame(columns = sensorCols)
        self.regionDF = pd.DataFrame(columns = regionCols)
        self.guessDF = pd.DataFrame(columns = guessCols)
        
    def initializeRegions(self, numPartitions, xMin, xMax, yMin, yMax, minAreaBound = 0.0):
        '''
        Update region list
        '''
        # Create polygon partition
        myPartition = PolygonPartition(numPartitions, xMin, xMax, yMin, yMax, 
                                       minAreaBound = minAreaBound)
        self.regions = myPartition.regions
        
    
    def initializeAgent_O(self, numOpaque, searchAngle, FOR, acquisitionDistance,stdDev=0.5):
        ''' 
        Initialize all Agent_O objects and sensor data 
        '''
        for aRegion in self.regions:
            if stdDev == "Uniform":
                pts = aRegion.getRandomPoints(numOpaque,distribution="Uniform")
            else:
                pts = aRegion.getRandomPoints(numOpaque,distribution="Normal", param=stdDev)
            agents = []
            for aLoc in pts:
                # Create agent
                myAgentO = Agent(aLoc, searchAngle, FOR, acquisitionDistance, self.agentIDCount)
                # Create list of agents
                agents.append(myAgentO)
                self.updateAgentODF(myAgentO, aRegion)
                self.agentIDCount += 1
            self.agent_O[aRegion] = agents
            
    
    def initializeAgent_T(self, numTransparent, searchAngle, FOR, acquisitionDistance,
                          acquisitionProbability=1.0, speed=1.0, angleError = 0.0,
                          boundingBox=[0,0,0,0]):
        ''' 
        Initialize Agent_T object and sensor data
        '''
        self.agent_T = []
        for i in range(numTransparent):
            myAgent = Agent(None, searchAngle, FOR, acquisitionDistance, self.agentIDCount, 
                            acquisitionProbability)
            myAgent.initLocation(boundingBox)
            myAgent.updateSpeedDirection(speed, myAgent.direction, error = angleError)
            self.agent_T.append(myAgent)
            self.agentIDCount += 1
    
    def findMinCOM(self, aLoc):
        '''
        Compute the region that minimizes the distance from aLoc
        to the region's center of mass.  Return the region and the distance.
        '''
        minDistance = 1e16
        minRegion = None
        for aRegion in self.regions:
            # Compute squared distance between aLoc and center of mass
            sqDist = hf.getDistanceSqrd(aRegion.centerOfMass, aLoc)
            if sqDist<minDistance:
                # If less than minDistance, update minDistance and minRegion
                minDistance = sqDist
                minRegion = aRegion
        return minRegion, sqrt(minDistance)
    
    def updateEstimatedLocations(self, anAgent, tolerance=0.0):
        '''
        Update estimated location for acquired and store appropriately.
        If location is within tolerance to a region's center of mass, set
        the region's center of mass to inactive.
        '''
        # Get estimated location
        estimatedLocation = anAgent.getEstimatedLocation()
        if not estimatedLocation is None:
            # Determine region with minimal distance to center of mass
            minRegion, minDistance = self.findMinCOM(estimatedLocation)
            # Update Guess dataframe
            self.updateGuessDF(deepcopy(anAgent),estimatedLocation, 
                               minDistance, minRegion.idNumber, 
                               anAgent.acqTarget)
            # If distance less than prescribed value, set region to inactive
            # and update dataframes
            if minDistance<self.guessTolerance:
                minRegion.isActive = False
                # Update AgentODF dataframe
                for anAgent_O in self.agent_O[minRegion]:
                    self.updateAgentODF(deepcopy(anAgent_O), minRegion)
                
           
    def updateTimeStep(self, dt):
        '''
        Update a time step for each agent
        '''
        # Update current time
        self.currentTime += dt
        # Update location
        for aAgentT in self.agent_T:
            aAgentT.updateAgentLocation(dt)
            # Acquire sensor targets
            aAgentT.updateAgentAcquisition(self.agent_O)
            # Update guess
            self.updateEstimatedLocations(aAgentT)
            # Update Data Frame
            self.updateAgentTDF(deepcopy(aAgentT))
            self.updateSensorDF(deepcopy(aAgentT))
        
                
    def updateAgentTDF(self, aAgent_T):
        '''
        Update location DF
        '''
        newRow = {"AgentID": aAgent_T.idNumber, "Time": self.currentTime, 
                  "Location": aAgent_T.location,
                  "FOR": aAgent_T.FOR*180/pi, "SensorAngle": (aAgent_T.searchAngle*180/pi)%360.0,
                  "SensorRadius": aAgent_T.acquisitionDistance, 
                  "Speed": aAgent_T.speed, "Direction": (aAgent_T.direction*180/pi)%360.0}
        self.agentTDF=self.agentTDF.append(newRow, ignore_index=True)
    
    def updateAgentODF(self, aAgent_O, aRegion):
        '''
        Update agent_O DF
        '''
        newRow = {"AgentID": aAgent_O.idNumber, "Time": self.currentTime,
                  "Location": aAgent_O.location, "RegionID": aRegion.idNumber,
                  "Active": aRegion.isActive}
        self.agentODF=self.agentODF.append(newRow, ignore_index=True)
        
    def updateSensorDF(self, aAgent_T):
        '''
        Update sensor DF for all possible acquisitions
        '''
        for anAcq in aAgent_T.acqTarget:
            # Acquired target
            self.updateSensorDFSingle(aAgent_T, anAcq, True)
        for anAcq in aAgent_T.notAcqTarget:
            # Not acquired target
            self.updateSensorDFSingle(aAgent_T, anAcq, False)
            
    def updateSensorDFSingle(self, aSensingAgent, aTgtAgent, isAcquired):
        '''
        Update sensor DF for a single possible acquisition
        '''
        newRow = {"AgentID": aSensingAgent.idNumber, "Time": self.currentTime,
                  "TgtID": aTgtAgent.idNumber, "TgtLocation": aTgtAgent.location,
                  "Acquired": isAcquired}
        self.sensorDF=self.sensorDF.append(newRow, ignore_index=True)
    
    def updateGuessDF(self, aAgent, guessLoc, minDist, minRegionID, acqAgentList):
        '''
        Update guess DF
        '''
        # Create list of IDs
        IDs = []
        for anAgent in acqAgentList:
            IDs.append(anAgent.idNumber)
        # Update DataFrame
        newRow = {"AgentID": aAgent.idNumber, "Time": self.currentTime, 
                  "Location": guessLoc, "MinDistance": minDist, 
                  "MinRegionID": minRegionID, "GuessIDs": IDs}
        self.guessDF = self.guessDF.append(newRow, ignore_index=True)
        
    def createRegionDF(self):
        '''
        Update region DF
        '''
        # Compute best guess (based on average)
        for aRegion in self.regions:
            myLoc = []
            for anAgent in self.agent_O[aRegion]:
                myLoc.append(anAgent.location)
            avgLoc = hf.locAverage(np.array(myLoc))
            errorLoc = sqrt(hf.getDistanceSqrd(avgLoc,aRegion.centerOfMass))
            newRow = {"RegionID": aRegion.idNumber, "Vertices": aRegion.vertices,
                      "Area": aRegion.area, "CenterOfMass": aRegion.centerOfMass,
                      "BestGuessLocation": avgLoc, "BestGuessError": errorLoc}
            self.regionDF=self.regionDF.append(newRow, ignore_index=True)
            

def saveData(myData, filePathStr = "", run=0):
    '''
    Save DataFrames to file located at filePathStr
    '''
    agentTDFName = filePathStr+"agentT_run"+str(run)+".json"
    myData.agentTDF.to_json(agentTDFName)
    agentODFName = filePathStr+"agentO_run"+str(run)+".json"
    myData.agentODF.to_json(agentODFName)
    sensorDFName = filePathStr+"sensor_run"+str(run)+".json"
    myData.sensorDF.to_json(sensorDFName)
    regionDFName = filePathStr+"region_run"+str(run)+".json"
    myData.regionDF.to_json(regionDFName)
    guessDFName = filePathStr+"guess_run"+str(run)+".json"
    myData.guessDF.to_json(guessDFName)

def checkDirectory(filePathStr=""):
    '''
    Check if directory exists.  If not, create directory for log files.
    '''
    if not os.path.isdir(filePathStr):
        os.mkdir(filePathStr)
        
##TEST
def main():
    # Initialize parameters from CONFIG.ini
    #Read CONFIG.ini file
    config_obj = configparser.ConfigParser()
    config_obj.read(CONFIG_FILE_NAME)
    runData = config_obj['RunInfo_Parameters']
    domainData = config_obj['Domain_Parameters']
    timeData = config_obj['Time_Parameters']
    opaqueData = config_obj['OpaqueAgent_Parameters']
    transparentData = config_obj['TransparentAgent_Parameters']
    guessData = config_obj['Guess_Parameters']
    
    # Domain Parameters
    xMin = float(domainData['xMin'])
    xMax = float(domainData['xMax'])
    yMin = float(domainData['yMin'])
    yMax = float(domainData['yMax'])
    numPartitions = int(domainData['numPartitions'])
    minAreaBound = float(domainData['minAreaBound'])
    # Time Parameters
    maxTime = float(timeData['MaxTime'])
    dt = float(timeData['timeStep'])
    # Opaque Agent Parameters
    numOpaque = int(opaqueData['numAgents'])
    searchAngle_O = float(opaqueData['searchAngle'])*pi/180.0
    FOR_O = float(opaqueData['FOR'])*pi/180.0
    stdDev = opaqueData['stdDeviation']
    if not stdDev == "Uniform":
        stdDev=float(stdDev)
    # Transparent Agent Parameters
    numTransparent = int(transparentData['numAgents'])
    print(numTransparent)
    searchAngle_T = float(transparentData['searchAngle'])*pi/180.0
    FOR_T = float(transparentData['FOR'])*pi/180.0
    acquisitionDistance = float(transparentData['acquisitionDistance'])
    acquisitionProbability = float(transparentData['acquisitioinProbability']) # Probability of Agent_T acquiring Agent_O when in sensor
    speed = float(transparentData['speed'])
    angleError = float(transparentData['moveAngleError'])*pi/180.0
    # Data save parameters
    filePath = runData['filePath']
    numRuns = int(runData['numRuns'])
    checkDirectory(filePathStr=filePath)
    # Guess Parameters
    guessTolerance = float(guessData['guessTolerance'])
    
    # TEMPORARY PARAMETERS
    # TODO: Make part of code based on initial parameters
    for aRun in range(numRuns):
        # Initialize global info (domain and agents)
        myData = GlobalInfo(guessTolerance)
        myData.initializeRegions(numPartitions, xMin, xMax, yMin, yMax, minAreaBound)
        myData.initializeAgent_O(numOpaque, searchAngle_O, FOR_O, 
                                 acquisitionDistance, stdDev=stdDev)
        myData.initializeAgent_T(numTransparent, searchAngle_T, FOR_T, 
                                 acquisitionDistance, acquisitionProbability, speed,
                                 angleError, [xMin, xMax, yMin, yMax])
        myData.createRegionDF()
        for aAgentT in myData.agent_T:
            myData.updateAgentTDF(deepcopy(aAgentT))
        
        # Update in time
        while myData.currentTime<maxTime:
            # Update data
            myData.updateTimeStep(dt)
        
        # Save dataframe
        saveData(myData, filePathStr=filePath, run=aRun)
    # Shut down logger
    logging.shutdown()   
        
if __name__ == '__main__':
    main()