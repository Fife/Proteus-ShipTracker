# Proteus: The vessel tracking software
My submission for the SynMax Twitter Programming contest in August 2021.
This is a program that takes in raw trackig data for many vessels over the course of a long period of time and determines the voyages for each vessel.
It also uses a Machine Learning model to predict the next three voyages for each of the ships found in the data. 

## To Run the program:

1) Install the dependancies listed in requirements.txt

2) Put Proteus.py, trident.py, and chart.py in the same directory along with tracking.csv, and ports.csv

3) Run Proteus.py, the outputs "voyages.csv" and "predict.csv" should be generated in the same folder.
