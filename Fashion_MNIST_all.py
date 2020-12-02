import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from collections import OrderedDict
from collections import namedtuple
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=90)
import pytorch_lr_finder as lr_finder




# Training set with original values and normalized ones
# for later testing purposes

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

loader = DataLoader(train_set, batch_size=1000, num_workers=1)
num_pixels = len(train_set) * 28 * 28

total_sum = 0
for batch in loader:
  total_sum += batch[0].sum()

mean = total_sum / num_pixels

sum_of_squared_error = 0
for batch in loader:
  sum_of_squared_error += (mean - batch[0]).pow(2).sum()

std = torch.sqrt(sum_of_squared_error / num_pixels)

train_set_normal = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
)

train_sets = {
    'not_normal' : train_set,
    'normal' : train_set_normal
}

# Initializing LeNet model

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.relu = nn.ReLU()
    self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5),
                           stride=(1,1), padding=(2,2))
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),
                           stride=(1,1), padding=(0,0))
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5),
                           stride=(1,1), padding=(0,0))
    self.linear1 = nn.Linear(in_features=120, out_features=84)
    self.linear2 = nn.Linear(in_features=84, out_features=10)

  def forward(self, t):
    t = self.relu(self.conv1(t))
    t = self.pool(t)
    t = self.relu(self.conv2(t))
    t = self.pool(t)
    t = self.relu(self.conv3(t))
    t = t.reshape(t.shape[0], -1)
    t = self.relu(self.linear1(t))
    t = self.linear2(t)

    return t

# application of learning rate finder

train_loader = DataLoader(
          train_set,
          batch_size=64,
          shuffle=True
    )

model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
lrf = lr_finder.LearningRateFinder(model, criterion, optimizer)
lrf.fit(train_loader)
lrf.plot()

# classes needed to visualize networks performance in tensorboard using
# different groups of hyperparameters

class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        
        return runs

class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader):
        
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        
        self.tb.add_image('images', grid)
        self.tb.add_graph(
            self.network,
            images.to(getattr(run, 'device', 'cpu'))
            )
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
    
    def end_epoch(self):
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        clear_output(wait=True)
        display(df)
        
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
        
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, fileName):
        
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

# Training loop 

params = OrderedDict(
    lr=[1e-2],
    batch_size=[1000],
    shuffle=[True],
    num_workers=[1],
    device=['cuda'],
    trainset=['normal','not_normal']
)

m = RunManager()
for run in RunBuilder.get_runs(params):
    
    device = torch.device(run.device)
    network = LeNet().to(device)
    loader = DataLoader(
        train_sets[run.trainset],
        batch_size = run.batch_size,
        shuffle = run.shuffle,
        num_workers = run.num_workers
    )
    optimizer = optim.Adam(
        network.parameters(), lr=run.lr
    )
    
    m.begin_run(run, network, loader)
    for epoch in range(15):
        m.begin_epoch()
        for batch in loader:
            
            images = batch[0].to(device)
            labels = batch[1].to(device)
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            m.track_loss(loss)
            m.track_num_correct(preds, labels)
            
        m.end_epoch()
    m.end_run()
m.save('results')

# tensorboard activation

tensorboard --logdir=runs

# building a confusion matrix

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
    return all_preds

prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
train_preds = get_all_preds(network, train_loader)

stacked = torch.stack(
    (
        train_set.targets,
        train_preds.argmax(dim=1)
    ),
    dim=1
)

conf_mt = torch.zeros(10,10, dtype=torch.int32)

for p in stacked:
    true_lab, pred_lab = p.tolist()
    conf_mt[true_lab, pred_lab] = conf_mt[true_lab, pred_lab] + 1

# plotting a confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                          plt.text(j, i, format(cm[i, j], fmt), 
                          horizontalalignment="center",
                          color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(10,10))
plot_confusion_matrix(conf_mt, train_set.classes)