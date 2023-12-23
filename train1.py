def train1(model,train_loader,criterion,optimizer):
  # model = CNN()
  model.train() # GAT = GATv2() # GAT.train()
  running_loss = 0.0
  # for i in range(len(train1_indc)):
  # Iterate in batches over the train1ing dataset.
  # print("t1")
  for i, data in enumerate(train_loader, 0):
      # print("t2")
      # if i ==3: # exit()
      inputs, labels,gf,idx = data
      # Zero the parameter gradients
      optimizer.zero_grad()
      # inputs=np.asarray(inputs,dtype=np.float32)
      # np.array(data, dtype=np.float32)

      # print("gf") # Forward pass: compute the outputs from the inputs

      labels = labels.view(-1,1)
      outputs = model(inputs,gf.float()) # Compute the loss between the outputs and the labels
      outputs = outputs.view(-1,1)
      loss = criterion(outputs.float(), labels.float()) # Backward pass: compute the gradients from the loss
      # print("SHAPE: ", outputs.shape, labels.shape)
      # exit()
      # optimizer.zero_grad()
      loss.backward() # Update the model parameters using the optimizer
      optimizer.step() # Accumulate the loss for printing
      running_loss += loss.item() # # Print statistics every 2000 mini-batches #
      # if i % 2000 == 1999: # # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
      # print('Finished Training')
  print("LOSS: ",running_loss)
def test1(model, test_loader, criterion,mx):
  #,mx):
  model.eval()
  total = 0.0
  correct = 0.0
  total_batch=0.0
  loss = 0.0
  for data in test_loader:
    # Getting the inputs and labels from the data loader
    inputs, labels,gf, idx = data # Forward pass: compute the outputs from the inputs
    # print("gff")
    outputs = model(inputs,gf.float()) # Get the predicted class by finding the index of the maximum value in each output vector
    # _, predicted = torch.max(outputs.float(), 1) # # Update the total number of predictions
    outputs = outputs.view(-1,1)
    labels = labels.view(-1,1)
    total += labels.size(0)
    print("OUTPUTS: ", outputs)
    print("---------------------")
    print( outputs*mx, idx)
    # print(outputs, labels,"---------------------")
    print("LABEL: ",labels)
    print("---------------------")
    print(labels * mx)
    sub = (outputs - labels)
    z = ((sub > 0.1) + (sub < -0.1)).sum() 
    print("ACC: ", z)
    correct += z.item()
    # exit()
    print("----------------------------------")
    # print(labels.shape)
    # print(outputs.shape)
    loss += criterion(outputs,labels.float())
    total_batch += 1
    
  print("LOSS TEST: ",loss,"================================================================================================================")
  print("CORRECT: ", 1- correct/total)
  # exit()
  return loss/total_batch, 1- correct/total