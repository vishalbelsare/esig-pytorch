if __name__ == "__main__":
    
    from tosig_pytorch import EsigPyTorch
    from torch.autograd import Function
    import torch
    import time
    import torch.nn as nn
    import torchvision
    from torch.utils.data.dataset import Dataset
    import pickle

    # use GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Autograd function
    class SigFn(Function):
        def __init__(self, m):
            super(SigFn, self).__init__()
            self.m = m
        def forward(self, X):
            sigs = EsigPyTorch(device).stream2sig(X, self.m)
            self.save_for_backward(sigs)
            return sigs
        def backward(self, grad_output):
            sigs, = self.saved_tensors
            result = sigs.backward(grad_output)
            return result

    def Sig(X,m):
        return SigFn(m)(X)

    class EsigNN001(torch.nn.Module):
        def __init__(self, signal_dimension, hidden_size, num_classes, signature_degree=3):
            super(EsigNN001, self).__init__()
            self.signature_degree = signature_degree
            self.sigdim = EsigPyTorch().sigdim(signal_dimension, signature_degree)
            self.linear1 = torch.nn.Linear(self.sigdim, hidden_size)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(hidden_size, num_classes)

        def esig_batches_processing(self, streams):
            sigs = [Sig(s, self.signature_degree) for s in streams]
            return torch.stack(sigs)

        def forward(self, streams):
            """
            In the forward function we accept a Tensor of input streams of data and we must return
            a Tensor of output data. This is the simplest example of a NN using signature as 1st layer.
            """
            out = self.esig_batches_processing(streams) 
            #out = Sig(streams, self.signature_degree)

            out = self.linear1(out)
            out = self.relu(out)
            out = self.linear2(out)
            return out

    # load streams
    with open('input.pickle', 'rb') as handle:
        X = pickle.load(handle, encoding='latin1')

    X = [torch.from_numpy(x) for x in X]
    X = [x.type(torch.double) for x in X]

    # load corresponding digit
    with open('output.pickle', 'rb') as handle:
        y = pickle.load(handle, encoding='latin1')

    class SequentialMNIST(torch.utils.data.dataset.Dataset):
        def __init__(self, X, y):
            self.streams = X
            self.labels = y

        def __getitem__(self, index):
            single_image_label = self.labels[index]
            img_as_tensor = self.streams[index]
            return (img_as_tensor, single_image_label)

        def __len__(self):
            return len(self.streams)

    # Hyper-parameters 
    signal_dimension = 2
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 1
    learning_rate = 0.01
    signature_degree = 3

    # Split train-test (naive way)
    split_1 = int(0.8 * len(X))
    X_train = X[:split_1]
    y_train = y[:split_1]
    X_test = X[split_1:]
    y_test = y[split_1:]

    # Sequential MNIST
    training_set = SequentialMNIST(X=X_train, y=y_train)
    testing_set = SequentialMNIST(X=X_test, y=y_test)

    # Custom collate_fn for batch processing
    def my_collate(batch):
        data = [item[0] for item in batch]
        max_length = max([d.shape[0] for d in data])
        new_data = []
        for d in data:
            if d.shape[0] == max_length:
                new_data.append(d)
            if d.shape[0] < max_length:
                d2 = torch.stack((max_length-d.shape[0])*[d[-1,:]], 0)
                d = d.view(d.size(0), -1)
                d2 = d2.view(d2.size(0), -1)
                new_d = torch.cat([d, d2], dim=0)
                new_data.append(new_d)
        new_data = torch.stack(new_data, 0)
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return new_data, target

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                               batch_size=batch_size,
                                               collate_fn=my_collate,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=testing_set, 
                                              batch_size=batch_size,
                                              collate_fn=my_collate,
                                              shuffle=False)

    #trainiter = iter(train_loader)
    #imgs, labels = trainiter.next()

    # Define the model
    model = EsigNN001(signal_dimension, hidden_size, num_classes, signature_degree)
    model = model.double()
    model.to(device)

    # Print learnable parameters
    for n, p in model.named_parameters():
        print(n, p.shape)


    # if multiple GPU are available
    model = nn.DataParallel(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start = time.time()
    # Train the model
    total_step = len(train_loader)
    epoch_loss = {}
    for epoch in range(num_epochs):
        losses = []
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        epoch_loss[epoch] = losses

    end = time.time()
    print('Total training time = {}'.format(end-start))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
