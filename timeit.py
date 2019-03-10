from Dataset import Dataset,Dataset_Original
import time

'''Show the execution time for the old and new generative methods'''

start = time.clock()
ds =  Dataset.generate()
print("New dataset creation speed: {0}".format(time.clock() - start))

start = time.clock()
df = Dataset_Original.generate()
print("Old dataset creation speed: {0}".format(time.clock() - start))
