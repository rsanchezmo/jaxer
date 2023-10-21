from dataset import Dataset
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

dataset = Dataset('./data/BTCUSD.csv', seq_len=20)
train_dataset, test_dataset = dataset.get_train_test_split(test_size=0.1)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Plot the data
data = next(iter(train_dataloader))
print(data)

plt.style.use('ggplot')
plt.figure(figsize=(14, 8))
plt.plot(data[0][0][:, 1], label='Close Price', color='blue')
plt.scatter(len(data[0][0][:, 1]), data[1][0], label='Next Close Price', color='red')

plt.title('BTC Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

