import numpy as np
import models
import datasets
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, train_loader, epochs, lr, wt_decay, loss_fn=torch.nn.CrossEntropyLoss()):
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_decay)

  training_hist = []
  for _ in tqdm(range(epochs)):
    total_loss = 0
    for (x,y) in train_loader:
      pred = model(x)
      loss = loss_fn(pred, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    training_hist.append(total_loss)
  
  plt.plot(np.arange(epochs), training_hist)

  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for (x,y) in train_loader:
      out = model(x)
      pred = torch.argmax(out, dim=1)
      correct += torch.sum(pred==y)
      total += x.shape[0]

  print('Final Train Accuracy', (100*correct)/total)

def test(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for (x,y) in test_loader:
      out = model(x)
      pred = torch.argmax(out, dim=1)
      correct += torch.sum(pred==y)
      total += x.shape[0]

  print('Final Test Accuracy', (100*correct)/total)

def main():
  train_ds = datasets.DFTFolderDataset('../datasets/PACS/photo')
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

  test_ds = datasets.DFTFolderDataset('../datasets/PACS/cartoon')
  test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=True)

  model = models.DFTModel(num_classes=len(train_ds.classes))

  train(model=model, train_loader=train_dl, epochs=100, lr=0.0001, wt_decay=0.00001)
  test(model=model, test_loader=test_dl)

if __name__ == '__main__':
  main()