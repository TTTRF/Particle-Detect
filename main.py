# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import random


class DTWKNN():
    """DTW-KNN for particle finding"""

    def __init__(self, neighbors=5, max_warping_window=10000, subsample_step=1):
        """neighbors: for every input sequence,the majority class of which defines the class the sequence belongs
           max_warping_window: the maximum difference between X and Y for counting DTW
           subsample_step:only part of input sequence is used to count DTW matrix"""
        self.neighbors = neighbors
        self.subsample_step = subsample_step
        self.max_warping_window = max_warping_window
        self.data = None
        self.label = None

    def input(self, data, label):
        """data: input training sequence,np array,shape(numbers of sequences,length of sequence)
           label:input training y for data,np array,shape(number of sequences,1)"""
        self.data = data
        self.label = label

    def _dtw_distance(self, ta, tb, d=lambda x, y: abs(x - y)):
        '''ta:time points for sequence A
           tb:time points for sequence B
           d:DTW-distance between two time points in different sequences
           it is counted using dynamic programming'''
        ta, tb = np.array(ta), np.array(tb)
        M, N = len(ta), len(tb)
        cost = sys.maxsize * np.ones((M, N))
        cost[0, 0] = d(ta[0], tb[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ta[i], tb[0])
        for j in range(1, N):
            cost[0, j] = cost[j - 1, 0] + d(ta[0], tb[j])  # deal with points on the boundaries
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                           min(N, i + self.max_warping_window)):  # must be counted within warping window
                cost[i, j] = min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]) + d(ta[i], tb[j])
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        """X:training sequence
           Y:testing sequence"""
        count = 0
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.zeros((x_s[0], y_s[0]), dtype='int')
        for i in range(0, x_s[0]):
            for j in range(0, y_s[0]):
                dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step], y[j, ::self.subsample_step])
                count = count + 1
        return dm, count

    def predict(self, test_data, test_data_num):
        '''
        test_data:input test X,shape(number of test_data,length of sequence)
        test_data_num: events & channels of test_data,np array [event,channel]
        '''
        dm, count = self._dist_matrix(test_data, self.data)
        knn_idx = dm.argsort()[:, :self.neighbors]
        knn_labels = self.label[knn_idx]  # five nearst neighbors
        max_wavelength = len(test_data_num)
        first_channel = test_data_num[0, 0]
        last_channel = test_data_num[-1, 0]
        max_channel = last_channel - first_channel + 1
        prediction_label = np.zeros(max_channel, dtype='int')
        current = 0
        for i in range(first_channel, last_channel + 1):
            count = 0
            while test_data_num[current, 0] == i:
                count = count + 1
                current = current + 1
                if current == max_wavelength:
                    break
            channel_labels = knn_labels[current - count:current, :]  # channels of an event
            channel_vote = np.sum(channel_labels, axis=1) >= 3  # which particle the channel is most likely to be
            if channel_vote.sum() >= count / 2:  # voting
                prediction_label[i - first_channel] = 1
            else:
                prediction_label[i - first_channel] = 0
        return first_channel, last_channel, prediction_label


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    tr = h5py.File('train.h5')
    # Waveform = tr['Waveform'][:]
    ParticleTruth = tr['ParticleTruth'][:]
    # wave_example = np.zeros((3000, 1031), dtype='int')
    # wave_test = np.zeros((4000, 1031), dtype='int')
    # for s in range(0, 3000):
    #     wave_example[s][0] = Waveform[s][0]
    #     wave_example[s][1] = Waveform[s][1]
    #     wave_example[s][2:1031] = Waveform[s][2]
    #     s = s + 1
    # for s in range(0, 4000):
    #     wave_test[s][0] = Waveform[s + 3000][0]
    #     wave_test[s][1] = Waveform[s + 3000][1]
    #     wave_test[s][2:1031] = Waveform[s + 3000][2]
    #     s = s + 1
    # np.save('wave_example.npy', wave_example)
    # np.save('wave_test.npy', wave_test)
    wave_example = np.load('wave_example.npy')
    wave_test = np.load('wave_test.npy')
    X_train = wave_example[0:3000:30, 2:-1]
    X_train_num = wave_example[0:3000:30, 0:2]
    max_wavelength = len(X_train)
    Y_train = np.zeros((max_wavelength, 1), dtype='int')

    X_test = wave_test[0:4000:4, 2:-1]
    X_test_num = wave_test[0:4000:4, 0:2]
    train_first_channel = X_train_num[0, 0]
    train_last_channel = X_train_num[-1, 0]
    current = 0
    for i in range(train_first_channel, train_last_channel + 1):
        count = 0
        while X_train_num[current, 0] == i:
            count = count + 1
            current = current + 1
            if current == max_wavelength:
                break
        if count == 0:
            continue
        else:
            Y_train[current - count:current, :] = ParticleTruth[i][5]  ##get Y_train here
    m = DTWKNN(5, 8, 1)
    m.input(X_train, Y_train)
    first_channel, last_channel, prediction = m.predict(X_test, X_test_num)
    count = last_channel - first_channel + 1
    accuracy = 0
    for idx in range(first_channel, last_channel + 1):
        real = ParticleTruth[idx][5]  ##get Y_test here
        print("通道：", idx, "   预测结果为：", prediction[idx - first_channel], "实际结果为:", real)
        if prediction[idx - first_channel] == real:
            accuracy += 1 / count
    print('分类正确率为： %.2f%%' % (accuracy * 100))
