import model

dataiter = iter(model.testloader)
images, labels = dataiter.next()

#print images 

model.imshow(model.torchvision.utils.make_grid(images))
print('Groundtruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
