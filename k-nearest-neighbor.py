import numpy as np
import operator
import math
import csv

#fungsi untuk mengambil data dari file yang sudah ada
def load_data(file_name):
    return np.genfromtxt(file_name, delimiter=',', skip_header=1)

#fungsi untuk menyimpan data ke dalam file csv
def save_data(data):
    data_file = open('output.csv', 'wt', newline ='')
    header = ['No','X1','X2','X3','X4','X5','Y']
    with data_file:
        writer = csv.writer(data_file, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)

#fungsi untuk menghitung jarak antara dua data dengan euclidean distance
def euclidean_distance(data1, data2):
    distance = 0
    for x in range(1,6):
        distance += pow((data2[x] - data1[x]),2)
    return math.sqrt(distance)

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
        countY = {}

        for data in neighbors[x]:
            response = data_train[data[0]][6]

            if response in countY:
                countY[response] += 1
            else :
                countY[response] = 1
        
        #ambil nilai kelas yang terbanyak
        predict[x] = max(countY.items(), key=operator.itemgetter(1))[0]
    
    return predict
    
#load file dan simpan ke data_file
data_file = load_data('DataTrain_Tugas3_AI.csv')

#600 data yang sudah ada dijadikan data train
data_train = [data_file[x] for x in range(600)]

#200 data yang sudah ada dijadikan data test
data_test = [data_file[x] for x in range(600,800)]

#k awal untuk uji coba
k = 5

#k hasil uji coba
k_final = k

maks = 0

#uji mana k yang probabilitas kebenaran datanya paling tinggi
while k < 50:
    predict = knn(data_train, data_test, k)
    hasil = 0
    for x in range(len(data_test)):
        if (predict[x] == data_test[x][6]):
            hasil += 1
    if (hasil > maks):
        maks = hasil
        k_final = k
    k += 2

#load data train semuanya
data_train = load_data('DataTrain_Tugas3_AI.csv')

#load data test semuanya
data_test = load_data('DataTest_Tugas3_AI.csv')

#hitung hasil prediksi setiap data test
predict = knn(data_train, data_test, k_final)

#masukkan nilai Y yang sudah di dapatkan ke data test
for x in range(len(data_test)):
    data_test[x][6] = predict[x]

#simpan hasil ke dalam file csv
save_data(data_test)