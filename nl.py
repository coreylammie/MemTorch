import torch
from torch.autograd import Variable
import memtorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memtorch.utils import LoadMNIST
import numpy as np
import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))

if __name__ == '__main__':
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=32, validation=False)
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    reference_memristor = memtorch.bh.memristor.LinearIonDrift
    reference_memristor_params = {'time_series_resolution': 1e-4}
    memristor = reference_memristor(**reference_memristor_params)
    model = Net().to(device)
    model.load_state_dict(torch.load('trained_model.pt'), strict=False)
    patched_model = patch_model(copy.deepcopy(model),
                              memristor_model=reference_memristor,
                              memristor_model_params=reference_memristor_params,
                              module_parameters_to_patch=[torch.nn.Conv2d],
                              mapping_routine=naive_map,
                              transistor=False,
                              programming_routine=naive_program,
                              programming_routine_params={'rel_tol': 0.2})
    patched_model.tune_()
    print(test(patched_model, test_loader))


# epochs = 1
# train_loader, validation_loader, test_loader = LoadMNIST(batch_size=32, validation=False)
# model = Net().to(device)
# criterion = nn.CrossEntropyLoss()
# learning_rate = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# best_accuracy = 0
# for epoch in range(0, epochs):
#     print('Epoch: [%d]\t\t' % (epoch + 1), end='')
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(data.to(device))
#         loss = criterion(output, target.to(device))
#         loss.backward()
#         optimizer.step()
#
#     accuracy = test(model, test_loader)
#     print('%2.2f%%' % accuracy)
#     if accuracy > best_accuracy:
#         torch.save(model.state_dict(), 'trained_model.pt')
#         best_accuracy = accuracy
