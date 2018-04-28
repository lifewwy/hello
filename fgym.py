import csv
import numpy as np
# import matplotlib.pyplot as plt

class make:

    def __init__(self,bbb):

        self.i = 0
        filename = 'csv/' + bbb + '.csv'
        print(filename)

        self.Len = 5
        self.state_dim = self.Len + 1


        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = np.asarray(list(reader))
            data = data.astype(np.float)
            data = np.transpose(data)
            # print(data[0,0:5])
            self.dd = data
        csvfile.close()

    def reset(self):
        self.i = 0
        self.state = np.zeros((1,self.Len+1))
        # print(self.state)

        # 持仓状态  -1     0     1
        #         持空头  无仓位  持多头
        self.positionStatus = 0

        self.state[0,0:self.Len] = self.dd[0, self.i : self.i + self.Len]
        self.state[0,self.Len] = self.positionStatus

        # print(self.state[0,])
        # print(self.state[0,4])

        return self.state


    def step(self,action):
        # action :  1,   2,    3,     4,   5
        #          开多, 开空, 无动作, 平多, 平空

        self.i = self.i+1

        if self.positionStatus == 0:
            if action == 1:
                self.positionStatus = 1
                reward = self.dd[0, self.i + self.Len-1] - self.dd[0, self.i + self.Len-2]
            if action == 2:
                self.positionStatus = -1
                reward = -( self.dd[0, self.i + self.Len-1] - self.dd[0, self.i + self.Len-2])
            if action == 3 or action == 4 or action == 5 :
                self.positionStatus = 0
                reward = 0

        if self.positionStatus == -1:
            if action == 5:
                self.positionStatus = 0
                reward = 0
            if action == 1 or action == 2 or action == 3 or action == 4 :
                self.positionStatus = -1
                reward = -( self.dd[0, self.i + self.Len-1] - self.dd[0, self.i + self.Len-2])

        if self.positionStatus == 1:
            if action == 4:
                self.positionStatus = 0
                reward = 0
            if action == 1 or action == 2 or action == 3 or action == 5 :
                self.positionStatus = 1
                reward = self.dd[0, self.i + self.Len-1] - self.dd[0, self.i + self.Len-2]

        self.state[0,0:self.Len] = self.dd[0, self.i : self.i + self.Len]
        self.state[0,self.Len] = self.positionStatus
        #print(state)
        #print(self.dd[0, self.i + self.Len-1])

        #print(self.state)

        # reward = 1
        done = False

        return self.state, reward, done

'''
dd = abcd('cu')

state = dd.reset()
print(state[0,])

for i in range(5):
    nextstate, reward, done = dd.step(1)
    print(nextstate[0,])
    print(reward)
'''






