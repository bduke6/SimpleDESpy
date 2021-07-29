# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 07:23:56 2021

@author: michael.a.yereniuk
"""

import configparser

CONFIGFILENAME = "CONFIG.csv"

config = configparser.ConfigParser()

# Save Info
sectionName = 'RunInfo_Parameters'
config.add_section(sectionName)
config.set(sectionName, 'FilePath', 'Data/')
config.set(sectionName, 'numRuns', '2')
# Domain Parameters
sectionName = 'Domain_Parameters'
config.add_section(sectionName)
config.set(sectionName, 'xMin', '20.0')
config.set(sectionName, 'xMax', '60.0')
config.set(sectionName, 'yMin', '20.0')
config.set(sectionName, 'yMax', '60.0')
config.set(sectionName, 'numPartitions', '10')
config.set(sectionName, 'minAreaBound', '0.0')

# Time Parameters
sectionName = "Time_Parameters"
config.add_section(sectionName)
config.set(sectionName, 'maxTime', '25.0')
config.set(sectionName, 'timeStep', '0.25')

# Opaque Agent Parameters
sectionName = "OpaqueAgent_Parameters"
config.add_section(sectionName)
config.set(sectionName, 'numAgents', '30')
config.set(sectionName, 'searchAngle', '0.0')
config.set(sectionName, 'FOR', '45.0')
config.set(sectionName, 'stdDeviation', 'Uniform')

# Transparent Agent Parameters
sectionName = "TransparentAgent_Parameters"
config.add_section(sectionName)
config.set(sectionName, 'numAgents', '10')
config.set(sectionName, 'searchAngle', '0')
config.set(sectionName, 'FOR', '60.0')
config.set(sectionName, 'acquisitionDistance', '10.0')
config.set(sectionName, 'acquisitioinProbability', '0.5')
config.set(sectionName, 'speed', '2.5')
config.set(sectionName, 'moveAngleError', '10.0')

# Guess Parameters
sectionName = 'Guess_Parameters'
config.add_section(sectionName)
config.set(sectionName, 'guessTolerance', '10.0')

# Write the new structure to the new file
with open(CONFIGFILENAME,'w') as configfile:
    config.write(configfile)