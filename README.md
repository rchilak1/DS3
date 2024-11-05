# CS4372 Monkey Model Visualizer

app.py 
  - Serves as the streamlit backend

tuning.py
  - Finds the best model by trying designated parameters.
  - Stores values in parameter_tuning_results.csv
    
loader.py
  - Loads in parameter_tuning_results.csv as a dataframe so user can choose best parameters.
  - User inputs ideal parameters as arguments
  - Finetunes/retrains model and stores weights as best_model.keras

png files
  - Screenshots of plots and metrics




