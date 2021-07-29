# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 07:57:05 2021

@author: michael.a.yereniuk
"""

try:
    HELPERFUNCTIONS_if
except NameError:
    HELPERFUNCTIONS_if=0
    import numpy as np
    from copy import copy
    from math import atan2, pi
    from statistics import mean

#########################
## ANIMATION FUNCTIONS ##
    #########################
def createAnimation(locationList, color="Black", interval=20, markersize=2, fig=1):
    '''
    Animate list of locations in time.  Format locationList[time][agentIdx][coordinate]
    '''
    locPlt, = plt.plot([locationList[0][0][0]],[locationList[0][0][1]], 'o', color=color,
                       markersize = markersize)
    # Create animation using the animate() function
    myAnimation = animation.FuncAnimation(fig, animateLocations, frames=np.arange(0,len(locationList[0]),1),
                                      fargs = [locationList, locPlt], interval=interval, blit=True, repeat=True)
    fig.canvas.draw()
    
def animateLocations(i, locations, locPlt):
    #locations = data[0]
    #locPlt = data[1]
    xdata = []
    ydata = []
    for aLoc in locations:
        xdata.append(aLoc[i][0])
        ydata.append(aLoc[i][1])
    locPlt.set_data(xdata, ydata)
    return locPlt,

    
def create_random_points(num_points = 10, xMin=40.0,xMax=60.0,yMin=60.0,yMax=80.0):
    points = np.empty((0, 2), float)
    for i in range(num_points):
        x = np.round(np.random.uniform(xMin, xMax, 1), 2)
        y = np.round(np.random.uniform(yMin, yMax, 1), 2)
        points = np.append(points,np.array([[x[0],y[0]]]),axis=0)

    return(points)

def in_box(points, bounding_box, eps=0.0):
    return np.logical_and(np.logical_and(bounding_box[0]-eps <= points[:, 0],
                                         points[:, 0] <= bounding_box[1]+eps),
                          np.logical_and(bounding_box[2]-eps <= points[:, 1],
                                         points[:, 1] <= bounding_box[3]+eps))

##################################################
## Determine whether point in polygon functions ##
##################################################
def isPointInPolygon(vertexList, x, y):
        '''
        Determine whether point (x,y) is in polygon bounded by ordered
        vertices from vertexList.
        '''
        numCrosses = 0
        point0 = vertexList[-1]
        for idx in range(len(vertexList)):
            point1 = vertexList[idx]
            if pointBelowLineSegment(point0, point1,x,y):
                numCrosses += 1
            point0 = vertexList[idx]
        if (numCrosses%2) == 0:
            return False
        else:
            return True
        
def pointBelowLineSegment(point0, point1, x, y):
    '''
    Determine whether point (x,y) intersects line segment
    bound by point0 and point 1
    '''
    # Make sure between x coordinates
    if (point0[0]<=x and point1[0]<=x) or (point0[0]>=x and point1[0]>=x):
        return False
    # Point above line segment's rectangle
    elif (point0[1]>=y and point1[1]>=y):
        return True
    # Point below line segment's rectangle
    elif (point0[1]<=y and point1[1]<=y):
        return False
    # Point inside line segment's rectangle
    m = (point1[1]-point0[1])/(point1[0]-point0[0])
    y_test = m*(x-point0[0])+point0[1]
    if y<y_test:
        return True
    else:
        return False

###################################
## Functions for Voronoi regions ##
###################################
def reflectPoints(myPoints, bounding_box):
    '''
    Reflect points across boundary.
    '''
    origPoints = copy(myPoints)
    for aPoint in origPoints:
        # Reflect across left boundary
        myPoints = np.append(myPoints,np.array([[2*bounding_box[0]-aPoint[0],aPoint[1]]]),axis=0)
        # Reflect across right boundary
        myPoints = np.append(myPoints,np.array([[2*bounding_box[1]-aPoint[0],aPoint[1]]]),axis=0)
        # Reflect across bottom boundary
        myPoints = np.append(myPoints,np.array([[aPoint[0],2*bounding_box[2]-aPoint[1]]]),axis=0)
        # Reflect across top boundary
        myPoints = np.append(myPoints,np.array([[aPoint[0],2*bounding_box[3]-aPoint[1]]]),axis=0)
    return myPoints

def filterRegions(voronoiObj, bounding_box, eps=0.0):
    '''
    Filter voronoi object and remove regions that have elements outside
    the bounding box. Also remove regions with no elements
    '''
    myRegions = copy(voronoiObj.regions)
    # Cull regions
    for aRegion in voronoiObj.regions:
        if len(aRegion)==0:
            # If region is emplty, delete region
            myRegions.remove(aRegion)
        else:
            # If region outside bounding box, delete region
            if False in in_box(voronoiObj.vertices[aRegion,:], bounding_box, eps):
                myRegions.remove(aRegion)
    return myRegions

######################
## Polygon Measures ##
######################
def areaPolygon(pointList):
    ''' 
    Compute area of polygon from list of points.  Points must be
    oriented either clockwise or counterclockwise. Note, if points are
    clockwise, then area will be negative. 
    
    '''
    point0 = pointList[-1]
    sum = 0
    for idx in range(len(pointList)):
        point1 = pointList[idx]
        sum += point0[0]*point1[1]-point1[0]*point0[1]
        point0 = pointList[idx]
    return sum/2

def centerOfMassPolygon(pointList):
    ''' 
    Compute center of mass from a list of polygon vertices. Uses formula
    for first moment in x,y coordinates. 
    
    '''
    # Need to pivot so error does not accumulate
    pivot = pointList[0]
    newList = [(aLoc[0]-pivot[0],aLoc[1]-pivot[1]) for aLoc in pointList]
    # Area required for formula
    A = areaPolygon(newList)
    sumX, sumY = 0, 0
    point0 = newList[-1]
    # Compute first moment summations
    for idx in range(len(newList)):
        point1 = newList[idx]
        sumX += (point0[0]+point1[0])*(point0[0]*point1[1]-point1[0]*point0[1])
        sumY += (point0[1]+point1[1])*(point0[0]*point1[1]-point1[0]*point0[1])
        point0 = newList[idx]
    centerX = sumX/(6*A)
    centerY = sumY/(6*A)
    # Return center of mass scaled by the pivot
    return (centerX+pivot[0], centerY+pivot[1])

#############################
## Measures from locations ##
#############################
def getDistanceSqrd(aLoc0, aLoc1):
    '''
    Get Euclidean squared distance between two 2-d locations
    '''
    return (aLoc0[0]-aLoc1[0])**2 + (aLoc0[1]-aLoc1[1])**2

def getAngle(aLoc0, aLoc1):
    '''
    Get angle from aLoc0 to aLoc1
    '''
    return atan2(aLoc1[1]-aLoc0[1],aLoc1[0]-aLoc0[0])

def locAverage(locList):
    '''
    Get average location from a list of locations
    '''
    x = mean(locList[:,0])
    y = mean(locList[:,1])
    return np.array([x,y])

##########################
## Measures from Angles ##
##########################
def angleDifference(angle1, angle2):
    ''' Compute the difference between 2 angles (in degrees) '''
    difference = angle1-angle2
    if difference>pi:
        difference = difference-2*pi
    elif difference<-pi:
        difference = 2*pi+difference
    return difference