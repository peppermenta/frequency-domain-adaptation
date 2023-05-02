import numpy as np
import models
import torchvision.datasets
import datasets
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs, lr, wt_decay, lr_decay, loss_fn=torch.nn.CrossEntropyLoss()):
  if wt_decay>0:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_decay)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
  lmbda = lambda epoch: lr_decay
  scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    

  training_hist = []
  best_acc = 0
  best_weights = None
  for _ in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for (x,y) in train_loader:
      x = x.to('cuda')
      y = y.to('cuda')
      out = model(x)
      loss = loss_fn(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      pred = torch.argmax(out, dim=1)
      correct += torch.sum(pred==y)
      total += x.shape[0]
      total_loss += loss.item()

      model.layer1.clamp(min=0.0)
      model.layer2_00.clamp(min=0.0)
      model.layer2_01.clamp(min=0.0)
      model.layer2_10.clamp(min=0.0)
      model.layer2_11.clamp(min=0.0)
      model.layer3_00.clamp(min=0.0)
      model.layer3_01.clamp(min=0.0)
      model.layer3_02.clamp(min=0.0)
      model.layer3_10.clamp(min=0.0)
      model.layer3_11.clamp(min=0.0)
      model.layer3_12.clamp(min=0.0)
      model.layer3_20.clamp(min=0.0)
      model.layer3_21.clamp(min=0.0)
      model.layer3_22.clamp(min=0.0)


    training_hist.append(total_loss)
    print('Train Loss', total_loss/len(train_loader.dataset))
    print('Train Accuracy', ((100*correct)/total).item())

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for (x,y) in val_loader:
        x = x.to('cuda')
        y = y.to('cuda')
        out = model(x)
        pred = torch.argmax(out, dim=1)
        correct += torch.sum(pred==y)
        total += x.shape[0]

    val_acc = (100*correct)/total
    if val_acc > best_acc:
      best_acc = val_acc.item()
      best_weights = model.state_dict()
    print('Validation Accuracy', val_acc.item())
    
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for (x,y) in train_loader:
      x = x.to('cuda')
      y = y.to('cuda')
      out = model(x)
      pred = torch.argmax(out, dim=1)
      correct += torch.sum(pred==y)
      total += x.shape[0]
  final_train_acc = (100*correct)/total
    
  print('Best Validation Accuracy', best_acc)
  print('Final Train Accuracy', final_train_acc.item())

def main():
  np.random.seed(0)
  torch.random.manual_seed(0)
  gen = torch.Generator().manual_seed(0)
  batch_size = 32
    
  ds = datasets.DFTFolderDataset('../datasets/PACS/sketch')
  train_ds, val_ds = torch.utils.data.random_split(ds, [0.75, 0.25], generator=gen)
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

  model = models.DFTModel(num_classes=len(ds.classes))
  model = model.to('cuda')
  model.load_state_dict(torch.load('../trained_models/lr0.001_decay0.001_lrdecay0.99_epochs100_acc68.10551452636719.pt'))

  train(model=model, train_loader=train_dl, val_loader=val_dl, epochs=20, lr=1e-3, wt_decay=1e-3, lr_decay=0.99)

if __name__ == '__main__':
  main()
