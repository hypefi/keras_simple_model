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

correct = 0
total = 0

#since we're not training, we don't need to calculate the gradient for our outputs

with model.torch.no_grad():
    for data in model.testloader:
        images, labels = data
        # calculate outputs by running images through the network 
        outputs = net(images)
        #the class with the highest energy is what we choose as prediction 
        _, predicted = model.torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
