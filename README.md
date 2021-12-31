# phaseClass
Classify seismic phases based on systematic move-out using an RNN

# Description

    Many to one sequence prediction:

    - detect phase arrivals based on systematic moveout 
      (label = 1 for seismic phase; label = 0 incorrect picks )
    - the provided example data consists of a 6 station array 
      that recorded 100 seismic phase with consistent move-outs
      and 100 erroneous picks
    - the recurrent neutral network is trained on synthetic data 
      for the specific array geometry

# Tutorial

  **5a_phase_create.py:** - create training data for the specific seismic network and expected earthquake locations,
                         for now only a simple homogenous velocity model is implemented but this can easily be extended
                         to 1D layered velocities using Obspy's tauP implementaiton
                         
  **5b_phase_detect.py**: - model training and performance evaluation including loss and validation curves as well as confusion
                            matrix
                          - RNN architecture (7 layers): LSTM(relu), Dropout, LSTM(relu), Dropout, LSTM(relu), Dropout, Dense(sigmoid)
                            all LSTM units are bi-directional
                          - accuracy on training and validation data reaches close to 100% see: **plots/5c_eq_phase_detect.png**

