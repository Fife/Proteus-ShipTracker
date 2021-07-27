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



print(allVoyages[allVoyages['begin_port_id']==90])
print(uniqueVoyages[uniqueVoyages['begin_port_id']==90])

