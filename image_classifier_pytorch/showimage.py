import matplotlib.pyplot as plt
import numpy as np

#functions to show image


def imshow(img):
    img = img/2 + 0.5 # unnormalize 
    npming = img.numpy()
    plt.imshow(np.transpose(npimg), (1,2,0))
    plt.show()

# get some random training images

dataiter = iter(trainloader)
images, labels = dataiter.next()


#show images
imshow(torchvision.utils.make_grid(images))

#print labels 
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


