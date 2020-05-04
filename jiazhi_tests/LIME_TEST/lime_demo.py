from lime import lime_image
import sys, os
import csv
import numpy as np
from keras.models import load_model
import keras.backend as K
from skimage.segmentation import slic
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = load_model('CNN.h5')

input_path = sys.argv[1]

isget = [0]*7
sample_data = []
class_data = []
test = []
with open(file = input_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:
        temp = []
        if(count<23000):            #23000
            count += 1
            continue
        if(isget[int(row[0])] == False):
            temp = []
            isget[int(row[0])] = True
            class_data.append(int(row[0]))
            row = row[1].split(' ')
            for i in range(len(row)):
                original = float(row[i])
                temp.append(0.333 * original)
                temp.append(0.333 * original)
                temp.append(0.333 * original)
                #row[i] = float(row[i])
            sample_data.append(temp)
            test.append(row)  
    csvfile.close()

sample_data = np.array(sample_data, dtype = float)
sample_data = np.reshape(sample_data, (7, 48, 48, 3))

test= np.reshape(test, (7, 48, 48, 1))
result = model.predict(test)
ans = []
for i in range(result.shape[0]):
    max = 0
    index = 0
    for j in range(result.shape[1]):
        if(result[i][j]>max):
            max = result[i][j]
            index = j
    ans.append(index)


# two functions that lime image explainer requires
def predict(input):
    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
    temp = input.transpose((3, 0, 1, 2))
    temp = np.reshape(temp[0], (10, 48, 48, 1))
    temp *= 3
    return model.predict(temp)
    # return ?

def segmentation(input):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    # return ?
    return slic(input, n_segments=50, sigma=5, compactness=100)

# Initiate explainer instance
explainer = lime_image.LimeImageExplainer()

for i in range(7):
    # Get the explaination of an image
    explaination = explainer.explain_instance(
                            image=sample_data[i], 
                            classifier_fn=predict,
                            segmentation_fn=segmentation,
                            num_samples=1000
                        )

    # Get processed image
    image, mask = explaination.get_image_and_mask(
                                label=ans[i],
                                positive_only=False,
                                hide_rest=False,
                                num_features=5,
                                min_weight=0.0
                            )

    # save the image
    # emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    image /= np.max(image)
    name = './' + 'fig3_' + str(class_data[i]) + '.jpg'
    plt.imsave(name, image)

    plt.figure()
    plt.imshow(image)
    plt.axis('on')
    plt.show()

