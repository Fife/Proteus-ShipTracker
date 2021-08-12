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
    nearest = tree.query([cartCoord], p=2, distance_upper_bound=10)
    index = nearest[1][0]
    #print(nearest[0][0])
    if nearest[0][0] > 3.5:
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
            #temp = temp.drop_duplicates(subset = 'end_port_id')
            #If nothing suitable is found in the global model, pick from most frequented destination in Vessel history
            if (dFGlobal[dFGlobal['begin_port_id'] == start_p].empty):
                guess = dF[dF['Mv'] == dF['Mv'].max()]
            else:
                #temp = dFGlobal[dFGlobal['begin_port_id'] == start_p]
                #Preference: By Global Weight - Worst Performance - ~27%
                #guess = temp[temp['GlobalWeight'] ==temp['GlobalWeight'].max()]
                
                #Preference: By Vessel Weight - Best Performance ~30%
                guess = dF[dF['guess'] == dF['guess'].max()]
                
                #Preference: By Global combined - In between ~28%
                #guess = temp[temp['guess'] ==temp['guess'].max()]
            
            if (guess['end_port_id'].size >1):
                randIndex = np.random.randint(guess['end_port_id'].size)
                guess_p = guess['end_port_id'].iloc[randIndex]
                voyage_id = guess['voyage_id'].iloc[randIndex]
            else:
                guess_p = guess['end_port_id'].iloc[0]
                voyage_id = guess['voyage_id'].iloc[0]
        #Else, pull data from suitable historical model
        else:
            guess = dF[(dF['vessel']== vessel) & (dF['begin_port_id']==start_p)]['guess'].max()
            guessF = dF[dF['guess'] == guess]
            guessF = guessF.drop_duplicates()
            guess_p = guessF['end_port_id'].iloc[0]
            voyage_id = guessF['voyage_id'].iloc[0]
            
        if (start_p == guess_p):
            guess = dFGlobal[(dFGlobal['begin_port_id']==start_p)]['GlobalWeight'].max()
            guessF = dFGlobal[dFGlobal['GlobalWeight'] == guess]
            guessF = guessF.drop_duplicates()
            if (guessF.empty):
                guess = dFGlobal[(dFGlobal['vessel']!=vessel)]['Mv'].max()
                guessF = dFGlobal[dFGlobal['Mv'] == guess]
                guessF = guessF.drop_duplicates()
            guess_p = guessF['end_port_id'].iloc[0]
            voyage_id = guessF['voyage_id'].iloc[0]

        start_p = guess_p
        guessFrame = guessFrame.append(pd.DataFrame({'vessel' : [vessel] ,
                                'begin_port_id' : [lastStart],
                                'end_port_id'  : [guess_p],
                                'voyage' : [x+1],
                                'voyage_id' : [voyage_id]}))
    return guessFrame


def writeVesselFrame(filtered):
    #Construct DataFrame from filtered results
    voyageFrame = pd.DataFrame({
        'vessel' : filtered.loc[filtered['portDiff']!=0, 'vessel'].astype(int),
        'begin_date': filtered.loc[filtered['portDiff']<0, 'datetime'],
        'end_date': filtered.loc[filtered['portDiff']>0, 'datetime'],
        'begin_port_id' : filtered.loc[filtered['portDiff']<0, 'portDiff'].astype(int),
        'end_port_id' : filtered.loc[filtered['portDiff']>0, 'portDiff'].astype(int)
        })

        
    #Do Some Cleanup, there may be a better way to construct the dataframe such that
    #there is less cleanup that needs to be done. Better construction also means that less
    #memory is taken up. This works for now, but if there is time try to clean it up better. 
    print("Performing Cleanup...")
    voyageFrame['begin_port_id'] = voyageFrame['begin_port_id'].abs()
    voyageFrame['end_port_id'] = voyageFrame['end_port_id'].shift(-1)
    voyageFrame['end_date'] = voyageFrame['end_date'].shift(-1)

    voyageFrame = voyageFrame.dropna()

    #print(voyageFrame[voyageFrame['vessel'] == 176])
    voyageFrame = voyageFrame[voyageFrame['begin_port_id'] != voyageFrame['end_port_id']]
        
    #The port ids get somehow assigned to float (maybe abs()?) so the id's must be cast to int
    voyageFrame['begin_port_id'] = voyageFrame['begin_port_id'].apply(np.int64)

    voyageFrame['end_port_id'] = voyageFrame['end_port_id'].apply(np.int64)
        
    #Sort Values by date
    voyageFrame = voyageFrame.sort_values(by = ['begin_date'])
        
    #Label Voyage data
    voyageFrame['voyage'] = range(len(voyageFrame))
    voyageFrame['voyage'] = voyageFrame['voyage'] +1
    voyageFrame = voyageFrame.reset_index(drop=True)
    
    #Write voyage data to .csv file
    print("Writing to file...")
    voyageFrame = voyageFrame.sort_values(by=['vessel'])
    answer = voyageFrame
    answer.drop(['voyage'], axis=1).to_csv(r'voyages.csv', index = False, header=True)
    return voyageFrame


def genFiltered(rawTrackingFrame, rawPortFrame):
    print("Sorting and Interpolating...")

    #Group by vessel and datetime
    sortedVesselFrame = rawTrackingFrame.sort_values(by=["vessel", "datetime"], ascending = True, ignore_index=True)

    #Sort ports by cartesean quadrent
    portFrame = rawPortFrame.sort_values(by=["lat","long"], ascending = True, ignore_index = True)

    #Interpolate NaN's Linearly by Group, and drop any remaining entries
    sortedVesselFrame = sortedVesselFrame.apply(lambda group: group.interpolate(method='linear', limit_area='inside'))
    sortedVesselFrame = sortedVesselFrame.dropna()

    #REMOVE --Keeping data down to 10 vessels for faster testing
    #sortedVesselFrame = sortedVesselFrame[sortedVesselFrame['vessel'] < 20]
    #END REMOVE

    print("Finding NN of ports for vessels...")

    threshold = 3
    #Create df from vessels whose speed is less than some threshold knots
    df = sortedVesselFrame[sortedVesselFrame['speed'] < threshold]

    #Find Nearest Port to the Vessels whose speeds are less than the threshold
    #Since, its worth checking if the Vessel is at port when speeds are very low
    #Rather than checking every single entry against every single port 
    #This reduces search time, but doesn't scale well with increasing ports or ships

    portTree = KDTree(portFrame)
    sortedVesselFrame['port']= df.apply(lambda x: findNearestPort(x['lat'], x['long'], portTree, portFrame), axis=1)

    #Filter out when the nearest port changes from >0 to 0 or NaN
    #0 Indicates Boat has stopped with no nearest port in search radius 
    print("Constructing Voyage Table...")
    sortedVesselFrame  = sortedVesselFrame.fillna(0)
    sortedVesselFrame['portDiff']=sortedVesselFrame['port'].diff()
    filtered = sortedVesselFrame[sortedVesselFrame['portDiff']!= 0]
    filtered = filtered.dropna()
    return filtered

def genRawTrain(voyageFrame):
    #Data Exploration
    #Unique voyages
    #allVoyages is a dataframe that contains all voyages along with a unique voyage speficier
    allVoyages = voyageFrame.sort_index()
    allVoyages = allVoyages.drop(['begin_date', 'end_date'], axis =1)
    allVoyages = allVoyages.sort_values(by=['begin_port_id', 'end_port_id'])

    #uniqueVoyages is a dataframe that contains all unique voyages, regardless of ship
    uniqueVoyages = allVoyages.drop(['vessel', 'voyage'], axis = 1)
    uniqueVoyages= uniqueVoyages.drop_duplicates()
    uniqueVoyages = uniqueVoyages.sort_values(by=['begin_port_id', 'end_port_id'])
    uniqueVoyages['voyage_id'] = range(len(uniqueVoyages))
    uniqueVoyages['vessel'] = allVoyages['vessel']

    #Finishing creating up allVoyages with unique data
    allVoyages['voyage_id']= uniqueVoyages['voyage_id']
    allVoyages['voyage_id'] = allVoyages['voyage_id'].fillna(method="ffill")
    allVoyages['voyage_id'] = allVoyages['voyage_id'].apply(np.int64)
    return allVoyages