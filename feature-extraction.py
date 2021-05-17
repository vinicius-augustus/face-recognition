# import libraries
from numpy import load
from numpy import asarray
from numpy import expand_dims
from numpy import savez_compressed
from numpy import reshape
from keras.models import load_model

# extract embeddings
def extract_embeddings(model, face_pixels):
    # convert data to float32
    face_pixels = face_pixels.astype('float32')
    # evaluate the mean of the data
    mean = face_pixels.mean()
    std = face_pixels.std()
    # evaluate the standard deviation of the data
    face_pixels = (face_pixels - mean)/std
    samples = expand_dims(face_pixels, axis=0)
    # expand the dimension of data
    yhat = model.predict(samples)
    return yhat[0]


# load the compressed dataset and facenet keras model
data = load('dataset.npz')
trainx, trainy = data['arr_0'], data['arr_1']
print(trainx.shape, trainy.shape)
model = load_model('facenet_keras.h5')

# get the face embeddings
new_trainx = list()
for train_pixels in trainx:
    embeddings = extract_embeddings(model, train_pixels)
    new_trainx.append(embeddings)
new_trainx = asarray(new_trainx)
# convert the embeddings into numpy array
print(new_trainx.shape)

# compress the 128 embeddings of each face
savez_compressed('dataset-embeddings.npz', new_trainx, trainy)
