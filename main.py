import tools

timer = tools.Timer()

# start counting the time spent by the program
timer.start()


debug = True  # choose smaller or full dataset

# training sample - 600 images
# test sample - 350 images

# training set - 42000 images
# test set - 28000 images


if debug:
    trainingFilesPath = './data/trainingSample/trainingSample/'
    testFilesPath = './data/testSample/testSample/'
else:
    trainingFilesPath = './data/trainingSet/trainingSet/'
    testFilesPath = './data/testSet/testSet/'


# find and load train images
trainData = tools.Data(trainingFilesPath)
trainData.load()

# init network and run training
network = tools.NeuralNetwork()
network.train(trainData.data, trainData.labels, 101, 0.0001, True)

# find and load test images
testData = tools.testData(testFilesPath)
testData.load()
testData.make_dirs()  # create directories for sorted images

network.predict(testData)

timer.stop()
timer.show()
