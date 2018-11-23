import numpy as np

def load_data():
    return np.genfromtxt('DataTrain_Tugas3_AI.csv', names=['index','X1','X2','X3','X4','X5','Y'], delimiter=',',skip_header=1)
    

data_file = load_data()

    