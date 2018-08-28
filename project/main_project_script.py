import cv2
import numpy as np
import functions
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn
from sklearn.datasets import fetch_mldata





if __name__ == '__main__':

    kernel = np.ones((3, 3), np.uint8)
    frameNumber = 0
    #classifier = functions.create_model((28, 28, 1), 10)
    #classifier.load_weights('cnnKerasWeights.h5')

    #Creating KNN model for number prediction
    # handle older versions of sklearn
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split

    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split

    # load the MNIST digits dataset
    #mnist = datasets.load_digits()

    ###
    custom_data_home = "./"
    mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
    print('MNIST Data Shape')
    print(mnist.data.shape)

    # take the MNIST data and construct the training and testing split, using 75% of the
    # data for training and 25% for testing
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                      mnist.target, test_size=0.25, random_state=42)

    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                    test_size=0.1, random_state=84)

    # show the sizes of each data split
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k
    kVals = range(1, 30, 2)
    accuracies = []

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 30, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)

        # evaluate the model and update the accuracies list
        score = model.score(valData, valLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))

    # re-train our classifier using the best k value and predict the labels of the
    # test data
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)


    predictions = model.predict(testData)

    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits
    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))

    videoNames = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi', 'video-5.avi',
                  'video-6.avi', 'video-7.avi', 'video-8.avi', 'video-9.avi']
    linesToPrint = []
    for video in videoNames:
        numbers = []
        id = -1
        suma = 0
        cap = cv2.VideoCapture(video)
        success, frame = cap.read()
        opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        line_points = functions.find_lines_on_frame(opening)
        print('LINE\n--------------------------------')
        print(line_points)
        print('--------------------------------')


        while success:
            frameNumber = frameNumber + 1
            success, frame = cap.read()
            if not success:
                break
            cv2.imshow('frame', frame)
            list_of_found_numbers = functions.find_numbers_on_image(frameNumber, frame, id)
            functions.detected_numbers(list_of_found_numbers, frameNumber, numbers, id)
            suma = functions.sum_numbers(numbers, frameNumber, line_points, suma, model)
            print(suma)

            if cv2.waitKey(1) == 13:
                break

        linesToPrint.append(video + ' ' + str(suma))
        cv2.destroyAllWindows()
        cap.release()

    with open('out.txt', 'a') as openedFile:
        openedFile.write('RA 55/2014 Arsenije Karpic\n')
        openedFile.write('file	sum\n')
        for line in linesToPrint:
            openedFile.write(line + '\n')
    openedFile.close()