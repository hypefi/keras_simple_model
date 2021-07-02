import model
import network

dataiter = iter(model.testloader)
images, labels = dataiter.next()
PATH = './cifar_net.pth'
#print images 

model.imshow(model.torchvision.utils.make_grid(images))
print('Groundtruth: ', ' '.join('%5s' % model.classes[labels[j]] for j in range(4)))

net = network.Net()
net.load_state_dict(model.torch.load(PATH))

outputs = net(images)

_, predicted = model.torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % model.classes[predicted[j]] for j in range(4)))

