import csv
import numpy as np

with open('csv/ru888.csv',newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = np.asarray(list(reader))
    data = data[0:6,1:5].astype(np.float)
    # data = np.transpose(data);
    o = np.array([x[0] for x in data])
    h = np.array([x[1] for x in data])
    l = np.array([x[2] for x in data])
    c = np.array([x[3] for x in data])
    print(o)
    print(h)
    print(l)
    print(c)
csvfile.close()
