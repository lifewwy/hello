import csv
import numpy as np
# from numpy import genfromtxt
import random

class abc:

    def __init__(self):
        self.k = 30 # k根bar的数据组成一个状态
        self.state_dim = self.k*4 + 3
        self.action_dim = 2

    def make(self,name):
        # name: 'ru', 'rb', 'sr'

        # self.ohlc = genfromtxt('csv/cu.csv', delimiter=',')  # ndarray

        # *.csv 只有一列
        with open('csv/dru.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = np.asarray(list(reader))
            data = data.astype(np.float)
            # data = np.transpose(data)
            self.ohlc = np.array([x[0] for x in data])
        csvfile.close()

        '''print('\n')
        # print(self.ohlc)
        for i in range(5):
            # print(i)
            print(self.ohlc[i])'''

    def reset(self):
        sn = random.randint(50,600)
        s = self.ohlc[sn*4-self.k*4:sn*4]
        s = np.array( list(s) + [0] +[0] +[1] )

        return sn, s

    def step(self,EntryBar,sn,action,flg):
        # sn: 状态序号，必须大于等于k；
        # flg: 持仓状态，1多头，-1空头，0空仓
        # action: 1平仓，0无动作

        r = self.ohlc[(sn+1)*4-1] - self.ohlc[sn*4-1]
        sn = sn + 1
        ss = self.ohlc[sn*4-self.k*4:sn*4]

        if action==1:
            flg = 0

        EntryPrice = self.ohlc[(EntryBar-1)*4]
        profit = ss[-1] - EntryPrice

        closePriceSinceEntry = self.ohlc[EntryBar*4-1:sn*4+1:4]
        # print( closePriceSinceEntry.shape )
        drawdown = closePriceSinceEntry.max() - closePriceSinceEntry.min()

        if (sn>=self.ohlc.size/4-5) or (drawdown>50000) or (flg==0):
            done = True # if ss is terminal state？
        else:
            done = False

        ss = np.array( list(ss)+[profit]+[drawdown]+[flg] )

        # return ss,profit,drawdown,flg,r,done
        return ss, r, done


'''p = abc()
p.make('ru')
ss, r, done = p.step(10,60,0,1)


print('\n')
print('ss')
for i in range(ss.size):
    # print(i)
    print(ss[i])

print('\n')
print(r)
print(done)
print(p.action_dim)'''
