# Image-Augmentation

Increasing Maize Leaf Disease Dataset Using the Contrast Technique

# Contributors

> The names below are Group 8 students (BBIT 4 Taita Taveta University, Academic year 2021/2022) who did a project on increase the maize leaf disease datasets. The supervisors name is **_Dr.Peter Ochieng_**.

- David Kinyanjui TU01-BE213-0400/2017
- Celestine Nyambiki Onenga TU01-BE213-0418/2017
- Lucy Gathoni TU01-BE213-0312/2017
- Njuguna Tilas TU01-BE213-0396/2017
- Peter Waithaka TU01-BE213-0034/2017
- Joram Muchiri TU01-BE213-0005/2015

## Code

### Imports

- Below are the packages and libraries that were essential for our project.

```
import PIL
import numpy as np
import os
import tensorflow as tf
import cv2
import keras
import pathlib
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy
```

### Loading the images

- The code below was used to import our maize leaf images.

```
data_dir=pathlib.Path(os.getcwd()+"/maize_leaf_diseases_dataset/")
maize_disease_dataset=list(glob.glob(f"{data_dir}/data/*.jpg"))
len(maize_disease_dataset)
```

### Converting the images into pixels

- The code below was used iterate through the entire list of paths to the images. Each image was converted into pixels which were later stored in a numpy array.

```
x=[]
for image in maize_disease_dataset:
    img=cv2.imread(str(image))#Convert image to pixels
    resized_img=cv2.resize(img,(180,180))#Images to be of same size
    x.append(resized_img)#Pixels
x=np.array(x) #Store pixels in numpy array
plt.imshow(x[0]) # Display the image using matplotlib
plt.axis('off')
plt.show()
```

### Training, Test Split

- In this section we splitted the datasets into test and training dataset. The test dataset was 30% and the training dataset was 70%.

```
x_train,x_test=train_test_split(x,test_size=0.3,random_state=0) #Splitting
print(len(x_train))
print(len(x_test))

# Scalling
x_train_scaled=x_train/255 #Scale of 0.,1.0
x_test_scaled=x_test/255
print(x_train_scaled[0])
plt.imshow(x_train_scaled[0]) #Display image
plt.axis('off')
plt.show()
```

### Creating our Augmentation 1 model

- In this section we created our CNN model, where we applied random contrast of 0.9, We then applied it on the training dataset. The augmentation images were saved in a local folder called Augmented.

```
augmentation1=keras.Sequential([
    tf.keras.layers.RandomContrast(
    0.9, seed=None
)
])
augmented_training1=augmentation1(x_train_scaled)
print(augmented_training1[0])
current_dir=pathlib.Path(os.getcwd())
# try:
if not os.path.isdir(str(current_dir)+"\Augmented"):
    os.mkdir("Augmented")
os.chdir(str(current_dir)+"\Augmented")
for i in range(len(augmented_training1)):
    plt.imsave(f"Augmented1_img{i}.jpg",augmented_training1[i].numpy())
# except:
#     pass
# finally:
os.chdir(str(current_dir))
plt.imshow(augmented_training1[0].numpy())
plt.axis('off')
plt.show()
```

### Creating our Augmentation 2 model

- In this section we created our CNN model 2, where we applied random contrast of 0.4, We then applied it on the training dataset. The augmentation images were saved in a local folder called Augmented.

```
augmentation2=Sequential([
    tf.keras.layers.RandomContrast(
    0.4, seed=None
    )
])
augmented_training2=augmentation2(x_train_scaled)
current_dir=pathlib.Path(os.getcwd())
try:
    if not os.path.isdir(str(current_dir)+"\Augmented"):
        os.mkdir("Augmented")
    os.chdir(str(current_dir)+"\Augmented")
    for i in range(len(augmented_training2)):
        plt.imsave(f"Augmented2_img{i}.jpg",augmented_training2[i].numpy())
except:
    pass
finally:
    os.chdir(str(current_dir))
plt.imshow(augmented_training2[0]) #Display our augmented image
plt.axis('off')
plt.show()
```

# Classification of the leaf maize images

- In this section we started loading our data that we used in the classification. The classification involved training a model to be able to identify whether an image is original or augmented.

```
augmented_labels_dict={
    "Original":0,
    "Augmented":1
}
maize_leaf_disease_dict={
    "Original":list(glob.glob(f"{data_dir}/data/*.jpg")),
    "Augmented":list(glob.glob(f"{os.getcwd()}/Augmented/*.png"))
}
print(len(maize_leaf_disease_dict["Original"]))
print(len(maize_leaf_disease_dict["Augmented"]))

```

# Converting the images into pixels

- In this section we converted the imported images into pixels and appended the pixels in a numpy array

```
x,y=[],[]
for augmented_original,images in maize_leaf_disease_dict.items():
    for image in images:
        img=cv2.imread(str(image))#Convert image to pixels
        resized_img=cv2.resize(img,(180,180))#Images to be of same size
        x.append(resized_img)#Pixels
        y.append(augmented_labels_dict[augmented_original])#Label Random 0,1
print(x[0])
print(y[:5])
x=np.array(x)
y=np.array(y)

```

### Splitting of training and test dataset

- We splitted the training and the test datasets in the ratio of 0.7 and 0.3

```
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0) #Splitting
print(len(x_train))
print(len(x_test))
print(x_train.dtype)
```

### Developed our CNN model

- In this section we created our CNN model that was supposed to help in classifying the augmented images and the originals

```
num_classes=2
model=Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),# Applied 16 filters of 3 X 3 Matrix
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    #128 neourons--> try and error
    layers.Dense(128,activation="relu"),
    layers.Dense(num_classes) #if 0 neuron is activated it means its Original, no activation(linear activation)
])
```

### Compiling of the model

- In this section we compiled our CNN model, we used adam as our optimizing factor

```
model.compile(optimizer="adam",
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"]
             )
```

### Training our model

- Fitting the training dataset and training it to identify the Original and augmented images

```
model.fit(x_train,y_train,epochs=10)
```

### Testing the model

- In this section we tested our model and found it to be 97.4% accurate

```
model.evaluate(x_test_scaled,y_test)
```

### Prediction

- In this section we gave our image some data to predict whether the image was augmented or original

```
prediction=model.predict(t_scale)
score=tf.nn.softmax(prediction[2]) #Picks the set of values in the tensor and picks the biggest on
print(score)
print(np.argmax(score)) #Returns the index of the max value
```

### Saving of the mode

- In this section we saved the model so that it can be used in other applications

```
prediction=model.predict(t_scale)
score=tf.nn.softmax(prediction[2]) #Picks the set of values in the tensor and picks the biggest on
print(score)
print(np.argmax(score)) #Returns the index of the max value
```
