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

#Create Filtered Subsets and training data
filteredFrame = trident.genFiltered(rawTrackingFrame, rawPortFrame)
rawTrainData = trident.genRawTrain(trident.writeVesselFrame(filteredFrame))
rawTrainData = rawTrainData.sort_values(by=['voyage'])
print("Generating ML Testing Data...")

rawTrain2 = rawTrainData.head(900)
rawTrain3 = rawTrainData.head(1000)
rawTrain4 = rawTrainData.head(1200)
rawTrain5 = rawTrainData.head(1400)
rawTrain6 = rawTrainData.head(1600)
rawTrain7 = rawTrainData.head(1800)
rawTrain8 = rawTrainData.head(2000)

answerKeyfinal = rawTrainData.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
testfinal = rawTrainData.drop(rawTrainData.groupby(['vessel']).tail(3).index, axis=0)

answerKey2 = rawTrain2.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test2 = rawTrain2.drop(rawTrain2.groupby(['vessel']).tail(3).index, axis=0)

answerKey3 = rawTrain3.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test3 = rawTrain3.drop(rawTrain3.groupby(['vessel']).tail(3).index, axis=0)

answerKey4 = rawTrain4.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test4 = rawTrain4.drop(rawTrain4.groupby(['vessel']).tail(3).index, axis=0)

answerKey5 = rawTrain5.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test5 = rawTrain5.drop(rawTrain5.groupby(['vessel']).tail(3).index, axis=0)

answerKey6 = rawTrain6.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test6 = rawTrain6.drop(rawTrain6.groupby(['vessel']).tail(3).index, axis=0)

answerKey7 = rawTrain7.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test7 = rawTrain7.drop(rawTrain7.groupby(['vessel']).tail(3).index, axis=0)

answerKey8 = rawTrain8.sort_values(['vessel', 'voyage']).groupby('vessel').tail(3)
test8 = rawTrain8.drop(rawTrain8.groupby(['vessel']).tail(3).index, axis=0)


#Use Scrubbed Data to create model
model = chart.DSHM(rawTrainData)

print("Training ML Model...")

model.train(test2, answerKey2, 1)
model.train(test3, answerKey3, 1)
model.train(test4, answerKey4, 1)
model.train(test5, answerKey5, 1) 
model.train(test6, answerKey6, 1)
model.train(test7, answerKey7, 1)
model.train(test8, answerKey8, 1)


model.train(testfinal, answerKeyfinal, 2)

print("Generating Final Prediction...")
finalGuess = model.predictPaths(3, False)
lastGuess = model.predictPaths(3, True)

#Write results out to csv file
finalGuess = finalGuess.sort_values(by=['vessel', 'voyage'])
finalGuess = finalGuess.reset_index(drop = True).drop(['voyage_id'], axis =1)
finalGuess.to_csv(r'predict.csv', index = False, header=True)

print(finalGuess)#[finalGuess['begin_port_id'] == finalGuess['end_port_id']]
