from scipy import spatial
import numpy as np

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