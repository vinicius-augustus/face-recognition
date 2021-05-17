# import libraries
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from os import listdir
from mtcnn.mtcnn import MTCNN

# extract Face
def extract_image(image):
    # open the image and convert to RGB format
    img1 = Image.open(image)
    img1 = img1.convert('RGB')
    # convert the image to numpy array
    pixels = asarray(img1)

    # start the MTCNN detector
    detector = MTCNN()  
    f = detector.detect_faces(pixels)

    # fetch the (x,y)co-ordinate and (width-->w, height-->h) of the image
    x1, y1, w, h = f[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1+w)
    y2 = abs(y1+h)
    # locate the co-ordinates of face in the image
    store_face = pixels[y1:y2, x1:x2]
    plt.imshow(store_face)

    # convert the numpy array to object
    image1 = Image.fromarray(store_face, 'RGB')
    # resize the image
    image1 = image1.resize((160, 160))
    # image to array
    face_array = asarray(image1)
    return face_array


# fetch the face
def load_faces(directory):
    face = []
    i = 1
    for filename in listdir(directory):
        path = directory + filename
        faces = extract_image(path)
        face.append(faces)
    return face


# get the array of face data(trainX) and it's labels(trainY)
def load_dataset(directory):
    x, y = [], []
    i = 1
    for subdir in listdir(directory):
        path = directory + subdir + '/'

        # load all faces in subdirectory
        faces = load_faces(path)

        # create labels
        labels = [subdir for _ in range(len(faces))]

        # summarize
        print("{} There are {} images in the class {}:".format(i, len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
        i = i+1
    return asarray(x), asarray(y)


# load the datasets
trainX, trainY = load_dataset('dataset/')
print(trainX.shape, trainY.shape)

# compress the data
savez_compressed('dataset.npz', trainX, trainY)
