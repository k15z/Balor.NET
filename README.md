# Balor.NET
Project **Balor** is my latest attempt at implementing a portable, lightweight computer vision and 
artificial intelligence library for the .NET framework. This code is probably never going to be 
production ready, but it is still going to be an awesome learning experience for me :).

## Example
The below code can be used to build a convolutional neural network for the MNIST handwriting data 
set; the first layers is a convolution layer which takes 28 by 28 greyscale images and applies 5 
different 4 by 4 filters. The next layer subsamples the 28 by 28 by 5 feature map, creating a 14 
by 14 by 5 feature map. The third layer flattens the feature map to a 1d array, and the final 
layer produces 10 output values indicating the digit shown in the image.

```
Convolution l1 = new Convolution(28, 28, 1, 4, 4, 5);
Subsampling l2 = new Subsampling(28, 28, 5, 2, 2);
DataFlatten l3 = new DataFlatten(14, 14, 5);
FeedForward l4 = new FeedForward(14*14*5, 10);

var OUTPUT = l4.feed(l3.feed(l2.feed(l1.feed(INPUT))))
l1.train(l2.train(l3.train(l4.train(EXPECTED - OUTPUT))))
```
