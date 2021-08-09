from scipy import spatial
import numpy as np
import pandas as pd

#Generalized formula for finding cartesian coords given lat and long
def cartesian(latitude, longitude):
    # Convert to radians
    latitude = latitude * (np.pi / 180)
    longitude = longitude * (np.pi / 180)

    R = 6371 # 6378137.0 + elevation  # relative to centre of the earth
    X = R * np.cos(latitude) * np.cos(longitude)
    Y = R * np.cos(latitude) * np.sin(longitude)
    Z = R * np.sin(latitude)
    return (X, Y, Z)


#Turn list of ports into list of cartesian coords for K-D Map
def KDTree(dataFrame):
    ports = []
    for index, row in dataFrame.iterrows():
        coordinate = [row['lat'], row['long']]
        cartCoord = cartesian(*coordinate)
        ports.append(cartCoord)
    return spatial.KDTree(ports)

#Find nearest port given a list of latitude, longitude, spatial map and dataFrame
#Returns 0 if vessel is slowed but away from port. 
def findNearestPort(lat, long, tree, dataFrame):
    cartCoord = cartesian(lat, long)
    nearest = tree.query([cartCoord], p=2)
    index = nearest[1][0]
    if nearest[0][0] > 50:
        return 0
    else:
        return dataFrame.port[index]
    
def generateGuesses(dF, vessel, start_p, numGuesses):
    dFGlobal = dF
    guessFrame = pd.DataFrame()
    dF = dF[dF['vessel']==vessel]
    tempList=[]
    tempList.append(vessel)
    for x in range(numGuesses):
        lastStart = start_p
        # If no suitable end port is found,pull data from global model
        if (dF[(dF['vessel']== vessel) & (dF['begin_port_id']==start_p)]['guess'].empty):
            temp = dFGlobal

            temp = temp.drop_duplicates(subset = 'end_port_id')
            temp = dFGlobal[dFGlobal['begin_port_id'] == start_p]
            guess = temp[temp['GlobalWeight'] ==temp['GlobalWeight'].max()]
            guess_p = guess['end_port_id'].iloc[0]
            start_p = guess_p
        #Else, pull data from suitable historical model
        else:
            guess = dF[(dF['vessel']== vessel) & (dF['begin_port_id']==start_p)]['guess'].max()
            guess_p = dF[dF['guess'] == guess].drop_duplicates()['end_port_id'].iloc[0]
            start_p = guess_p

        guessFrame = guessFrame.append(pd.DataFrame({'vessel' : [vessel] ,
                                'begin_port_id' : [lastStart],
                                'end_port_id'  : [guess_p],
                                'voyage' : [x+1] }))
    return guessFrame
