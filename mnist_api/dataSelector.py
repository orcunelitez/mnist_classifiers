
import random
import numpy as np

class dataSelector():
    def __init__(self, data, label):
        
        self.__dataset = {}
        for i in range(10):
            self.__dataset[i] = []

        for i in range(len(label)):

            self.__dataset[label[i]].append((data[i], label[i]))

        for i in range(10):
            random.shuffle(self.__dataset[i])

        

    def findSubset(self, numberOfElements):
        data = []
        labels = []
        for i in range(10):
          #  data.append(self.__dataset[i][:numberOfElements][:])
            data += self.__dataset[i][:numberOfElements]

        random.shuffle(data)

        #return data[:][0], data[:][1]
        
        labels = [item[-1] for item in data]
        data2 = [item[:-1] for item in data]
        return np.asarray(data2), labels



if __name__ == "__main__":
    dataS = dataSelector([1,2,3,4,5],[6,7,8,9,0])
    a, b = dataS.findSubset(3)




    

        

