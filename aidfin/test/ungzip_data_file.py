# -*- coding: UTF-8 -*-
import gzip, binascii, struct, numpy

import matplotlib
import matplotlib.pyplot as plt

with gzip.open("MNIST_data/t10k-images-idx3-ubyte.gz") as f:
    # Print the header fields.
    for field in ['magic number', 'image count', 'rows', 'columns']:
        # struct.unpack reads the binary data provided by f.read.
        # The format string '>i' decodes a big-endian integer, which
        # is the encoding of the data.
        print(field, struct.unpack('>i', f.read(4))[0])

    # Read the first 28x28 set of pixel values.
    # Each pixel is one byte, [0, 255], a uint8.
    buf = f.read(28 * 28)
    image = numpy.frombuffer(buf, dtype=numpy.uint8)

    # Print the first few values of image.
    print('First 10 pixels:', image[:10])


    # We'll show the image and its pixel value histogram side-by-side.
    _, (ax1, ax2) = plt.subplots(1, 2)

    # To interpret the values as a 28x28 image, we need to reshape
    # the numpy array, which is one dimensional.
    ax1.imshow(image.reshape(28, 28), cmap=plt.cm.Greys);

    ax2.hist(image, bins=20, range=[0, 255]);


    # Let's convert the uint8 image to 32 bit floats and rescale
    # the values to be centered around 0, between [-0.5, 0.5].
    #
    # We again plot the image and histogram to check that we
    # haven't mangled the data.
    scaled = image.astype(numpy.float32)
    scaled = (scaled - (255 / 2.0)) / 255
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(scaled.reshape(28, 28), cmap=plt.cm.Greys);
    ax2.hist(scaled, bins=20, range=[-0.5, 0.5]);
    plt.show()

    with gzip.open("MNIST_data/train-labels-idx1-ubyte.gz") as f:
        # Print the header fields.
        for field in ['magic number', 'label count']:
            print(field, struct.unpack('>i', f.read(4))[0])

        print('First label:', struct.unpack('B', f.read(1))[0])
