import trident
import chart


print("Imorting Data...")
#Input Files        
portFile = 'ports.csv'
trackingFile = 'tracking.csv'

#Output Files
voyageFile = 'voyages.csv'
predictFile = 'predict.csv'

#Read in data sets
rawTrackingFrame = trident.pd.read_csv(trackingFile)
rawPortFrame = trident.pd.read_csv(portFile)

filteredFrame = trident.genFiltered(rawTrackingFrame, rawPortFrame)
rawTrainData = trident.genRawTrain(trident.writeVesselFrame(filteredFrame))

#Use Scrubbed Data to create model

model = chart.DSHM(rawTrainData)

#Use built in functionality of model to predict n number of paths
finalGuess = model.predictPaths(3)

print(model.predictFrame)

#Write results out to csv file
finalGuess.reset_index(drop = True)
finalGuess.to_csv(r'predict.csv', index = False, header=True)

print(finalGuess)
