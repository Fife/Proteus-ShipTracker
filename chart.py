import trident

class DSHM:
    def __init__(self, allVoyages):
        self.allVoyageStats = allVoyages
        self.trainingData = trident.pd.DataFrame()
        self.vesselStats = trident.pd.DataFrame()
        self.globalWeight = trident.pd.DataFrame()
        self.histWeight = trident.pd.DataFrame()
        self.predictFrame = trident.pd.DataFrame()
        self.generateTrainData()
        self.generateWeightTables()
        self.generatePredict()


    def generateTrainData(self):
        #Implementing a ML Model Step 1: 
        #Setting up the training data.
        #Takes a statistical dataFrame as an input and generates appropriate training data
        #returns a dataframe of training data

        # # of times completed a voyage
        self.allVoyageStats = self.allVoyageStats.sort_values(by=['voyage_id'])
        self.allVoyageStats['global_voy_freq'] = self.allVoyageStats.groupby('voyage_id')['voyage_id'].transform('count')

        # # of total voyages out of port
        self.allVoyageStats['total_voy_out_port'] = self.allVoyageStats.groupby('begin_port_id')['begin_port_id'].transform('count')

        # # of times particular vessel completed voyage
        self.vesselStats = self.allVoyageStats.sort_values(by=['vessel', 'voyage_id'])
        self.vesselStats['times_compl_vessel'] = self.vesselStats.groupby(['voyage_id', 'vessel'])['voyage_id'].transform('count')

        # # of total voyages for vessel
        self.vesselStats['total_voy_vessel'] = self.vesselStats.groupby('vessel')['voyage_id'].transform('count')

        # # of endpoints for a given start point
        self.trainingData = self.vesselStats
        self.trainingData['num_endpoints'] = self.vesselStats.groupby(['begin_port_id'])['end_port_id'].transform('nunique')


        # In[10]:

    def generateWeightTables(self):
        #Implementing a ML Model Step 2: 
        #Aggrigate the Data into 2 weight tables
            # Global weight table 
            # Historical Vessel weight table
        self.globalWeight = self.trainingData.drop(['voyage','times_compl_vessel','total_voy_vessel', 'vessel'], axis =1).drop_duplicates()

        self.globalWeight['Mg/Ne'] = (self.globalWeight['global_voy_freq']/self.globalWeight['total_voy_out_port'])/self.globalWeight['num_endpoints']
        self.globalWeight['Mg/Ne'] = self.globalWeight['Mg/Ne']

        self.globalWeight = self.globalWeight.drop(['voyage_id'], axis=1).drop_duplicates()


        self.globalWeight['Sigma'] = self.globalWeight.groupby('begin_port_id')['Mg/Ne'].transform('sum')
        self.globalWeight['GlobalWeight'] = self.globalWeight['Mg/Ne']/self.globalWeight['Sigma']
        self.globalWeight = self.globalWeight.drop(['global_voy_freq','total_voy_out_port','num_endpoints'], axis =1)

        #print(globalWeight)#[globalWeight['begin_port_id']==27])

        self.histWeight = self.trainingData.drop(['voyage','total_voy_out_port','times_compl_vessel','global_voy_freq','num_endpoints' ], axis =1)

        self.histWeight['times_ended_at'] = self.histWeight.groupby(['vessel','end_port_id'])['vessel'].transform('count')
        self.histWeight['Mv'] = self.histWeight['times_ended_at']/self.histWeight['total_voy_vessel']
        #histWeight = histWeight.drop_duplicates()
        #print(histWeight[histWeight['vessel']==1])#.sort_index())

        # In[11]:

    def generatePredict(self):
        #Implementing a ML Model Step 3:
        #Write a function to easily generate guesses given the 2 weight table frames. 
        #Populate a new data frame containing predictions.
        self.predictFrame = self.histWeight
        self.predictFrame = self.predictFrame.sort_values(by=['vessel'])
        self.predictFrame = self.predictFrame.merge(self.globalWeight)

        self.predictFrame = self.predictFrame.drop(['Mg/Ne','Sigma','total_voy_vessel','times_ended_at',], axis =1)
        self.predictFrame

        self.predictFrame['guess'] = self.predictFrame['Mv']*self.predictFrame['GlobalWeight']

    def predictPaths(self, numPredicts):
        vesselList = self.histWeight.vessel.unique()
        vesselList = vesselList.tolist()
        beegGuess = trident.pd.DataFrame()
        for i in vesselList:
            lastKnown= self.histWeight[self.histWeight['vessel']==i].sort_index()
            last = lastKnown['end_port_id'].iloc[-1]
            beegGuess = beegGuess.append(trident.generateGuesses(self.predictFrame,i, last,numPredicts))
            beegGuess = beegGuess.drop_duplicates()
        finalGuess = beegGuess
        return finalGuess