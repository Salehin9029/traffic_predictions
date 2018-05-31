import csv
import numpy
filename = 'data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)
data = numpy.array(x)
print(data.shape)