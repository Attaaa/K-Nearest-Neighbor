import numpy as np
import operator

#fungsi untuk mengambil data dari file yang sudah ada
def load_data():
    return np.genfromtxt('DataTrain_Tugas3_AI.csv', delimiter=',', skip_header=1)
    
#fungsi untuk menghitung jarak antara dua data dengan euclidean distance
def euclidean_distance(data1, data2):
    distance = 0
    for x in range(1,6):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

#fungsi utama untuk algoritma knn
def knn(data_train, data_test, k):
    neighbors = {}

    #lakukan perhitungan sebanyak data test
    for x in range(len(data_test)):
        distances = {}

        #hitung jarak semua tetangga dari satu data test dengan semua data train
        for y in range(len(data_train)):
            distances[y] = (euclidean_distance(data_train[y], data_test[x]))

        #urutkan hasil jarak tetangga terdekat 
        distances = sorted(distances.items(), key=operator.itemgetter(1))

        #ambil tetangga dengan jarak terdekat sebanyak k
        neighbors[x] = [distances[i] for i in range(k)]

    predict = {}

    #hitung nilai kelas dari data tetangga yang di dapat
    for x in neighbors:
        classVotes = {}

        for data in neighbors[x]:
            response = data_train[data[0]][6]

            if response in classVotes:
                classVotes[response] += 1
            else :
                classVotes[response] = 1
        
        predict[x] = max(classVotes.items(), key=operator.itemgetter(1))[0]

    print(predict)

data_file = load_data()

# 500 data yang sudah ada dijadikan data train
data_train = [data_file[x] for x in range(500)]

# 300 data yang sudah ada dijadikan data test
data_test = [data_file[x] for x in range(500,800)]

knn(data_train, data_test, 5)