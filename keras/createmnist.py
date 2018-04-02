import numpy
import json
import codecs

#Do this for each of the npy files in the mnist npz (zip file https://s3.amazonaws.com/img-datasets/mnist.npz)
a = numpy.load("x_train.npy")
b = a.tolist()
json.dump(b, codecs.open("x_train.json", "w", encoding="utf-8"), separators=(',', ':'), sort_keys=True, indent=4)