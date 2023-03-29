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
      model.layer1 = torch.clamp(model.layer1, min=0.0)
      model.layer2_00 = torch.clamp(model.layer2_00, min=0.0)
      model.layer2_01 = torch.clamp(model.layer2_01, min=0.0)
      model.layer2_10 = torch.clamp(model.layer2_10, min=0.0)
      model.layer2_11 = torch.clamp(model.layer2_11, min=0.0)
      model.layer3_00 = torch.clamp(model.layer3_00, min=0.0)
      model.layer3_01 = torch.clamp(model.layer3_01, min=0.0)
      model.layer3_02 = torch.clamp(model.layer3_02, min=0.0)
      model.layer3_10 = torch.clamp(model.layer3_10, min=0.0)
      model.layer3_11 = torch.clamp(model.layer3_11, min=0.0)
      model.layer3_12 = torch.clamp(model.layer3_12, min=0.0)
      model.layer3_20 = torch.clamp(model.layer3_20, min=0.0)
      model.layer3_21 = torch.clamp(model.layer3_21, min=0.0)
      model.layer3_22 = torch.clamp(model.layer3_22, min=0.0)

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

  train(model=model, train_loader=train_dl, epochs=3, lr=0.001, wt_decay=0.0001)
  test(model=model, test_loader=test_dl)

if __name__ == '__main__':
  main()