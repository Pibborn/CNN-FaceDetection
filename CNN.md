[TOC]

#Introduction: Convolutional Neural Networks

A Convolutional Neural Network (CNN from here onwards) is a neural network with a "deep" architecture, i.e. with more than one hidden layer. CNNs are inspired by biological research on animals' visual system, which reveals that the cells in the visual cortex are arranged in a way so that each cell is sensible to a section of the animal's visible field [[1]]. For this reason they are expecially used in Computer Vision tasks.

[[1]]	Hubel, D. and Wiesel, T. (1968). Receptive fields and functional architecture of monkey striate cortex. Journal of Physiology (London), 195, 215–243.   

----------


##Architecture Overview

As it often is when talking about "deep learning", there is no precise architecture that perfectly defines all CNNs; however, some common characteristics can be found in all the work dealing with CNNs. 

- A **convolutional layer**: at least one of the hidden layers implements the signal processing operation of "convolution". This will be explained in detail later. A convolutional layer is not densely connected and typically has a nonlinear activation function for its neurons. The neurons in a convolutional layer are arranged in a 3D volume, reflecting the structure of the input. 
- A **pooling layer**: a pooling layer usually comes immediately after a convolutional layer: its purpose is to reduce the dimensionality of the output of the convolutional layer. This layer's activation function is a block function that takes a matrix as input and returns a single value, for example $max(matrix)$ or $avg(matrix)$. This operation is usually called *downsampling* in image processing.
- A **fully-connected layer**: after convolving and downsampling the input image the desired amount of times (by stacking layers), the neural net should have learned high level features. However, we still seek to make the final inference using these features; we could be interested in outputting class scores for a classification task [[2]], or pixel positions for regression tasks [[3]]. This is done by stacking one or more fully-connected layers on top of the existing architecture. 

Literature on CNNs also defines the neurons responsible for receiving the input as the *input layer* and the neurons on the deepest level as the *output layer*.

## The Image Domain

Before diving into the details of how CNNs work, it makes sense to introduce some image processing concepts to better understand why they are a good fit for Computer Vision tasks. 

###A Color Image is a 3D Matrix: the RGB model
An image can be thought of as a map of pixels: each pixel has a location $(x, y)$ and a color $c$. In the real world there are infinitely possible colors, but for digital images the accepted standard is the RGB color model. In this model, each pixel is colored by combining its Red, Green and Blue component: each component can have a value from 0 to 255 (8 bit for each "color channel", so 24 bits of *color depth* total).
This way, a color image can be thought of as being made up of three stacked matrices: one stores the Red component for each pixel, one stores the Green component and the last one stores the Blue component.

A greyscale image is a 2D matrix with each pixel having a value from 0 to 255; a black-and-white image is again 2D, but its pixel's color values can only be $0$ (black) or $1$ (white).

###A Color Image is a Discrete Function: Convolution

Convolution is a signal processing operation originally defined on continuous functions: it outputs a third function which shape depends on the correlation between the two input functions. Usually, the first input function $f$ is the original "signal" we are interested in processing; the second input function $g$ is a "convolution function", a special function that has no meaning on its own but is useful to modify the first function via convolution. This process is sometimes called *filtering*, as some convolution functions are useful to filter out unwanted characteristics of the image, or to enhance desirable ones.

An image can be thought of as a two-variable function $i(x, y) = c$, where c is the pixel color that can be computed as described in the preceding paragraph. $x, y$ and $c$ are discrete, finite values: they can be represented as a matrix. When computing convolution between an image matrix and a *convolution matrix*, the following computation is performed.

![convolution-image](http://i.imgur.com/aD2Cuiv.png)
*(from MACHINE VISION, Ramesh Jain, Rangachar Kasturi, Brian G. Schunck. McGraw-Hill 1995)*

As we can see in the figure, the convolution matrix $[A \dots F]$ is multiplied elementwise with a block of the image $[p1 \dots p9]$. These values are then summed to give the final convolution result for output image pixel $h[i, j]$. Then, the convolution matrix "slides" throughout the original image and the computation is performed again. In the end, the output will be another matrix that can be interpreted as an image.


![](http://cse19-iiith.vlabs.ac.in/neigh/convolution.jpg)
*(From http://cse19-iiith.vlabs.ac.in/theory.php?exp=neigh)*


Convolution can be seen as "superimposing" the convolution matrix over the image matrix. Note that each pixel of the original image gets its turn in being the "center" of the convolution operation. This way, the output image size is the same as the input image size. 

Some convolution matrices examples ($\star$ is the convolution operator):

![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) $ \star \begin{bmatrix}0 & 0 & 0\\0 & 1 & 0\\ 0 & 0 & 0\end{bmatrix} = $ ![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) (identity)


![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) $ \star \begin{bmatrix}1& 1 & 1\\1 & 1 & 1\\ 1 & 1 & 1\end{bmatrix} * 1/9 = $ ![blur](https://upload.wikimedia.org/wikipedia/commons/0/04/Vd-Blur2.png) (blurring)

![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) $ \star \begin{bmatrix}-1& -1 & -1\\-1 & 8 & -1\\ -1 & -1 & -1\end{bmatrix}= $ ![blur](https://upload.wikimedia.org/wikipedia/commons/6/6d/Vd-Edge3.png) (edge detection)

*(From [Wikipedia](https://en.wikipedia.org/wiki/Kernel_%28image_processing%29)*).

## The Convolutional Layer
As we saw in the last convolution example, a convolution matrix can be useful to extract some features from the image. This is the fundamental idea of the convolutional layer: the data coming in from the input layer should be convolved with a matrix that extracts meaningful features from it. So, the weights $w$ we want to learn are the values of the convolution matrix.

### Depth Columns: Local Connectivity
As convolution is a local operation, it is unnecessary for the convolution layer to be fully connected with the preceding layer. This helps in reducing the number of parameters that the net has to learn. 

![prova](http://cs231n.github.io/assets/cnn/depthcol.jpeg)

On the left, there is the *input layer*, taking in 32x32 pixels color pictures (so a 32x32x3 matrix); on the right, there is the *convolutional layer*, with a *depth column* of five neurons all looking at the same area in the picture. A convolutional layer is a *volume* of neurons: depth columns are useful to learn different features in the same region of the picture. The extent of the input area these neurons are connected with can be defined as a parameter of the layer, but is usually a square region. 

### Depth Slices: Parameter Sharing 
In the convolution volume/3D matrix $C$, a *depth slice* is the 2D matrix of neurons at a certain depth $d$: $C[:,:,d]$. To further reduce the number of parameters that the convolutional layer has to learn, we can have all the neurons at depth $d$ *share* the learned weights. 

Since we defined the weights as the values of the convolution matrix, this also makes sense from an image processing point of view: when we compute the convolution operation between an image and a convolution matrix, we keep the same convolution matrix while sliding throughout the image. This way, the input image gets processed the same way in all its areas, just as it is in classical image processing.

Summarizing depth columns and depth slices:

- Neurons in the same depth column ($C[x, y, :]$) look at the same area of the picture, but filter (convolute) differently. Therefore, the neurons in a depth column extract *different features from the same area*;
- Neurons in a depth slice ($C[:,:,d]$) implement the same filtering operation in all areas of the picture. Therefore, they extract *the same feature from the whole picture*.

###Activation Function
While any nonlinear activation function can be used, some work has shown the rectifier function $f(x)=max(0, x)$ to work better than $tanh$ or the sigmoid[[2]]. Another argument that could be made in favor of the ReLU function is that it better preserves the filtering work made by the convolution operation. 

###Output Volume
As we discussed, the convolution layer learns a series of filters, the weights in a depth slice. The image will be convolved with each one of these filters; therefore on top of being a volume the convolutional layer also outputs a volume. Its depth is equal to the depth of the convolutional layer, while its width and height depend on some parameters.

###Convolutional Layer Parameters

####Filter Size
Recall that each depth column is connected to a small area of the input layer. This can be changed to have a bigger or smaller convolution matrix, for any reason.

####Stride
Instead of sliding the convolution matrix pixel-by-pixel over the original image, we can make bigger "jumps": as a result, the output image gets smaller, and the filters we are learning have less overlap. If the stride is greater than half the filter size, the convolution layer does not cover the whole input area. We will not be learning any feature from these areas.

**todo: una bella immagine esplicativa**

####Number of Filters
Which is also the size of a depth column. We may interested in learning a lot of features from the image, or just one.

**todo: esempio grafico di come i parametri cambino le dimensioni dell'output volume**

##Pooling Layer
The pooling layer implements a downsampling operation, which could be useful if the output volume of the convolutional layer is too big. 

![max-pooling](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

Here, max-pooling is implemented: from each 2x2 block of each depth slice of the output volume the maximum value is taken. There are alternatives such as average pooling, but max-pooling is more used in practice.

Note that a pooling layer does not reduce the depth of the volume.

##Fully-connected layer
The fully-connected layer is a "classic" neural net layer. One interesting thing to note is that the last layer has to be one-dimensional, so it is responsible for "squashing" the three-dimensional convolution and pooling results into the final class prediction or regression score. 