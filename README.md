# CS4372 Assignment 3 Visualizer

app.py 
  - serves as the streamlit backend

tuning.py
  - finds the best model by trying designated parameters.
  - Stores values in parameter_tuning_results.csv
    
loader.py
  - loads in parameter_tuning_results.csv as a dataframe so user can choose best parameters.
  - User inputs ideal parameters as arguments
  - Finetunes/retrains model and stores weights as best_model.keras

epochInfo.png
  - shows training/validation accuracy and loss per epoch for best model

parameters.png
  - shows training/validation accuracy values for each tested parameter combination




