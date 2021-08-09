import pandas as pd 
import trident 

pd.set_option('display.max_columns', 500)


print("Imorting Data...")
#Input Files        
portFile = 'ports.csv'
trackingFile = 'tracking.csv'

#Output Files
voyageFile = 'voyages.csv'
predictFile = 'predict.csv'

#Read in data sets
rawTrackingFrame = pd.read_csv(trackingFile)
rawPortFrame = pd.read_csv(portFile)

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

threshold = 5
#Create df from vessels whose speed is less than some threshold knots
df = sortedVesselFrame[sortedVesselFrame['speed'] < threshold]

#Find Nearest Port to the Vessels whose speeds are less than the threshold
#Since, its worth checking if the Vessel is at port when speeds are very low
#Rather than checking every single entry against every single port 
#This reduces search time, but doesn't scale well with increasing ports or ships

portTree = trident.KDTree(portFrame)
sortedVesselFrame['port']= df.apply(lambda x: trident.findNearestPort(x['lat'], x['long'], portTree, portFrame), axis=1)

#Filter out when the nearest port changes from >0 to 0 or NaN
#0 Indicates Boat has stopped with no nearest port in search radius 
print("Constructing Voyage Table...")
sortedVesselFrame  = sortedVesselFrame.fillna(0)
sortedVesselFrame['portDiff']=sortedVesselFrame['port'].diff()
filtered = sortedVesselFrame[sortedVesselFrame['portDiff']!= 0]
filtered = filtered.dropna()

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
voyageFrame = voyageFrame[voyageFrame['begin_port_id'] != voyageFrame['end_port_id']]

#The port ids get somehow assigned to float (maybe abs()?) so the id's must be cast to int
voyageFrame['begin_port_id'] = voyageFrame['begin_port_id'].apply(trident.np.int64)
voyageFrame['end_port_id'] = voyageFrame['end_port_id'].apply(trident.np.int64)

#Sort Values by date
voyageFrame = voyageFrame.sort_values(by = ['begin_date'])

#Label Voyage data
voyageFrame['voyage'] = range(len(voyageFrame))
voyageFrame['voyage'] = voyageFrame['voyage'] +1
voyageFrame = voyageFrame.reset_index(drop=True)

#Write voyage data to .csv file
print("Writing to file...")
voyageFrame.to_csv (r'voyages.csv', index = False, header=True)

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
allVoyages['voyage_id'] = allVoyages['voyage_id'].apply(trident.np.int64)



#print(allVoyages[allVoyages['begin_port_id']==90])
#print(uniqueVoyages[uniqueVoyages['begin_port_id']==90])

# In[9]:


#Implementing a ML Model Step 1: 
#Setting up the training data

#2 or 3 Data Frames, still trying to figure out exactly what the model needs.
#Will probably edit this cell/portion multiple times


# # of times completed a voyage
allVoyageStats = allVoyages.sort_values(by=['voyage_id'])
allVoyageStats['global_voy_freq'] = allVoyageStats.groupby('voyage_id')['voyage_id'].transform('count')

# # of total voyages out of port
allVoyageStats['total_voy_out_port'] = allVoyageStats.groupby('begin_port_id')['begin_port_id'].transform('count')

# # of times particular vessel completed voyage
vesselStats = allVoyageStats.sort_values(by=['vessel', 'voyage_id'])
vesselStats['times_compl_vessel'] = vesselStats.groupby(['voyage_id', 'vessel'])['voyage_id'].transform('count')

# # of total voyages for vessel
vesselStats['total_voy_vessel'] = vesselStats.groupby('vessel')['voyage_id'].transform('count')

# # of endpoints for a given start point
trainingData = vesselStats
trainingData['num_endpoints'] = vesselStats.groupby(['begin_port_id'])['end_port_id'].transform('nunique')

#uniqueVoyages
#trainingData[trainingData['begin_port_id']==32]['vessel' == ].sort_values(by=['global_voy_freq'])


# In[10]:


#Implementing a ML Model Step 2: 
#Aggrigate the Data into 2 weight tables
    # Global weight table 
    # Historical Vessel weight table


globalWeight = trainingData.drop(['voyage','times_compl_vessel','total_voy_vessel', 'vessel'], axis =1).drop_duplicates()

globalWeight['Mg/Ne'] = (globalWeight['global_voy_freq']/globalWeight['total_voy_out_port'])/globalWeight['num_endpoints']
globalWeight['Mg/Ne'] = globalWeight['Mg/Ne']

globalWeight = globalWeight.drop(['voyage_id'], axis=1).drop_duplicates()


globalWeight['Sigma'] = globalWeight.groupby('begin_port_id')['Mg/Ne'].transform('sum')
globalWeight['GlobalWeight'] = globalWeight['Mg/Ne']/globalWeight['Sigma']
globalWeight = globalWeight.drop(['global_voy_freq','total_voy_out_port','num_endpoints'], axis =1)

print(globalWeight)#[globalWeight['begin_port_id']==27])

histWeight = trainingData.drop(['voyage','total_voy_out_port','times_compl_vessel','global_voy_freq','num_endpoints' ], axis =1)

histWeight['times_ended_at'] = histWeight.groupby(['vessel','end_port_id'])['vessel'].transform('count')
histWeight['Mv'] = histWeight['times_ended_at']/histWeight['total_voy_vessel']
#histWeight = histWeight.drop_duplicates()
print(histWeight[histWeight['vessel']==1])#.sort_index())


# In[11]:


#Implementing a ML Model Step 3:
#Write a function to easily generate guesses given the 2 weight table frames. 
#Populate a new data frame containing predictions.
predictFrame = histWeight
predictFrame = predictFrame.sort_values(by=['vessel'])
predictFrame = predictFrame.merge(globalWeight)

predictFrame = predictFrame.drop(['Mg/Ne','Sigma','total_voy_vessel','times_ended_at',], axis =1)
predictFrame

predictFrame['guess'] = predictFrame['Mv']*predictFrame['GlobalWeight']

#When we make a prediction:
    #Multiply all globals by corresponding Mv's, 
    #Any that don't exist for a given Vessel, get multiplied by 0
    #Sort by result, and pick the highest number

#If no history of endport in vessel, check for a suitable endport across all voyages
#A suitable endpoint would be a port_id that exists as an endport in another vessel history
#and also exists in the current vessel    

 
lastKnown= histWeight[histWeight['vessel']==1].sort_index()
vesselList = histWeight.vessel.unique()
vesselList = vesselList.tolist()
beegGuess = pd.DataFrame()
for i in vesselList:
    lastKnown= histWeight[histWeight['vessel']==i].sort_index()
    last = lastKnown['end_port_id'].iloc[-1]
    beegGuess = beegGuess.append(trident.generateGuesses(predictFrame,i, last,3))
    beegGuess = beegGuess.drop_duplicates()
finalGuess = beegGuess


# In[23]:

finalGuess.reset_index(drop = True)
finalGuess.to_csv(r'predict.csv', index = False, header=True)
#finalGuess


finalGuess.reset_index(drop = True)
finalGuess.to_csv(r'predict.csv', index = False, header=True)
#finalGuess
