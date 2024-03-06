# %%
# verification
print('--------')
print('pytorch-gpu testing...')
import torch
print('torch.cuda.is_available():' + 
    str(torch.cuda.is_available()))
print('--------')

if __name__ == '__main__':
    # https://zhuanlan.zhihu.com/p/35434175
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.autograd import Variable
    import time
    
    # print start time
    print("Start time = "+time.ctime())
    
    # read data
    inp = np.loadtxt("./test/input" , dtype=np.float32)
    
    oup = np.loadtxt("./test/output", dtype=np.float32)
    #inp = inp*[4,100,1,4,0.04,1]
    oup = oup*500
    inp = inp.astype(np.float32)
    oup = oup.astype(np.float32)
    # Hyper Parameters
    input_size = inp.shape[1]
    hidden_size = 1000
    output_size = 1
    num_epochs = 1000
    learning_rate = 0.001
    
    # Toy Dataset
    x_train = inp
    y_train = oup
    
    # Linear Regression Model
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Net, self).__init__()
            #self.fc1 = nn.Linear(input_size, hidden_size) 
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.l1 = nn.ReLU()
            self.l2 = nn.Sigmoid()
            self.l3 = nn.Tanh()
            self.l4 = nn.ELU()
            self.l5 = nn.Hardshrink()
            self.ln = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            out = self.fc1(x)
            out = self.l3(out)
            out = self.ln(out)
            out = self.l1(out)
            out = self.fc2(out)
            return out
    
    model = Net(input_size, hidden_size, output_size)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    
    ###### GPU
    if torch.cuda.is_available():
        print("We are using GPU now!!!")
        model = model.cuda()
    
    # Train the Model 
    for epoch in range(num_epochs):
        # Convert numpy array to torch Variable
        if torch.cuda.is_available():
            inputs  = Variable(torch.from_numpy(x_train).cuda())
            targets = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs  = Variable(torch.from_numpy(x_train))
            targets = Variable(torch.from_numpy(y_train))
    
        # Forward + Backward + Optimize
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print ('Epoch [%d/%d], Loss: %.8f' 
                   %(epoch+1, num_epochs, loss.item()))
    
    # print end time
    print("End time = "+time.ctime())
    
    # Plot the graph
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).data.cpu().numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    plt.plot( y_train/500, 'r-', label='Original data')
    plt.plot( predicted/500,'-', label='Fitted line')
    #plt.plot(y_train/500, predicted/500,'.', label='Fitted line')
    plt.legend()
    plt.show()
    
    # Save the Model
    torch.save(model.state_dict(), 'model.pkl')