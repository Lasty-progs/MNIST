import numpy
from PIL import Image, ImageOps
import pathlib
import time
import os


class Timer:
    '''A class for calculating the time spent on parts of the program'''
    def __init__(self):
        self.st = 0
        self.times = []

    def start(self):
        '''start calculating time'''
        self.st = time.time()

    def stop(self):
        '''save the elapsed time from the start'''
        self.times.append(time.time() - self.st)

    def show(self, index=-1):
        '''show chosed saved time(default = last)'''
        print(self.times[index])


class Data:
    '''A class for operations with train image data'''
    def __init__(self, mainDir):
        # Parsing folder and file names from project data
        self.__private_currentDirs = [
            file for file in pathlib.Path(mainDir).iterdir()if file.is_dir()]

    def to_numpy(self, im):
        '''fastest way to convert Pil image to numpy massive
        (source of this solve: https://habr.com/ru/articles/545850/)'''
        im.load()
        # unpack data
        e = Image._getencoder(im.mode, 'raw', im.mode)
        e.setimage(im.im)

        # NumPy buffer for the result
        shape, typestr = Image._conv_type_shape(im)
        data = numpy.empty(shape, dtype=numpy.dtype(typestr))
        mem = data.data.cast('B', (data.data.nbytes,))

        bufsize, s, offset = 65536, 0, 0
        while not s:
            l, s, d = e.encode(bufsize)
            mem[offset:offset + l] = d
            offset += l
        if s < 0:
            raise RuntimeError("encoder error %d in tobytes" % s)
        return data

    def load(self):
        '''load data and labels and prapare for training model '''
        # Downloading images to numpy array
        labels = []
        data = []

        for currentDir in self.__private_currentDirs:
            for currentFile in currentDir.iterdir():
                labels.append(int(currentDir.name))
                image = Image.open(currentFile)
                image.load()
                image = ImageOps.invert(image)
                image = self.to_numpy(image)
                image = image.flatten()
                data.append(image)

        self.labels = numpy.array(labels)
        self.data = numpy.array(data)


class testData(Data):
    '''A class for operations with test image data'''
    def __init__(self, mainDir):
        self.__private_currentDir = pathlib.Path(mainDir)

    def load(self):
        '''load data and prapare for predict by model '''
        data = []
        for currentFile in self.__private_currentDir.iterdir():
            image = Image.open(currentFile)
            image.load()
            image = ImageOps.invert(image)
            image = super().to_numpy(image)
            data.append(image)

        self.data = data

    def make_dirs(self):
        '''create folders for sorted by model data'''
        self.__private_mainPath = os.getcwd()
        try:
            os.mkdir(self.__private_mainPath + "/results")
        except FileExistsError:
            pass
        try:
            os.mkdir(self.__private_mainPath + "/results" + "/sorted_data")
            for i in range(10):
                os.mkdir(self.__private_mainPath +
                         "/results" + "/sorted_data/" + str(i))
        except FileExistsError:
            print("Error with creates folers outs")


class NeuralNetwork:
    '''A class for create, train and use model'''

    # Creating weight for model (784 in 10 out)
    def __init__(self, ins=784, outs=10):
        weights = []
        for i in range(outs):
            a = numpy.random.rand(ins)
            weights.append(a)
        weights = numpy.array(weights, dtype=numpy.float64)
        self.weights = weights

    def __private_sigmoid(self, x):  # Activate func
        '''simple sigmoid func'''
        return 1/(1 + numpy.exp(-x))

    def __private_trainPredict(self, input, weights):
        '''func for train model'''
        return self.__private_sigmoid(input.dot(weights))

    def predict(self, input):
        '''func for use model'''
        self.__private_mainPath = os.getcwd()
        for instance, name in zip(input.data, range(len(input.data))):
            out = []
            for i in range(10):
                out.append(self.__private_sigmoid(
                    instance.flatten().dot(self.weights[i])))

            instance = instance.astype(numpy.uint8)
            out_img = Image.fromarray(instance)
            out_img = ImageOps.invert(out_img)
            out_img.save(self.__private_mainPath + "/results" +
                         "/sorted_data/" + str(out.index(max(out))) +
                         "/" + str(name) + ".jpg", "JPEG")

    def train(self, data, labels, epochs=1, alpha=1, visualize_weights=False):
        '''train model func(if you want to save visualisation weights,
        set last flag to true)'''
        if visualize_weights:
            self.__private_weight_folders()

        for epoch in range(epochs):
            print("Epoch: " + str(epoch))
            for instance in range(len(labels)):

                goal = numpy.zeros(10)
                goal[labels[instance]] = 1

                result = 0
                for number in range(10):
                    result = self.__private_trainPredict(data[instance],
                                                         self.weights[number])
                    delta = result - goal[number]
                    # error = delta ** 2

                    weight_deltas = data[instance]*delta
                    self.weights[number] -= weight_deltas * alpha

            if visualize_weights and epoch % 100 == 0:  # will fix it

                self.__private_show_weighs(epoch)

    def __private_show_weighs(self, epoch):
        '''save visualized weights images'''
        for number in range(10):
            show_weights = numpy.zeros([28, 28])

            for i in range(28):
                show_weights[i] = (self.__private_sigmoid(
                    self.weights[number][28*i:28*(i+1)] * 2) * 256).astype(int)

            show_weights = show_weights.astype(numpy.uint8)
            out_img = Image.fromarray(show_weights)

            out_img.save(self.__private_mainPath + "/results" +
                         "/visualized_weights/" + str(number) +
                         "/" + str(epoch) + ".jpg", "JPEG")

    def __private_weight_folders(self):
        '''create folders for visualized weights'''
        self.__private_mainPath = os.getcwd()
        try:
            os.mkdir(self.__private_mainPath + "/results")
            os.mkdir(self.__private_mainPath +
                     "/results" + "/visualized_weights/")
            for i in range(10):
                os.mkdir(self.__private_mainPath +
                         "/results" + "/visualized_weights/" + str(i))
        except FileExistsError:
            print("weight dirs was exists")
