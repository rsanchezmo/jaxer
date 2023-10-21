import matplotlib.pyplot as plt
import pandas as pd

# Convert the 'Date' column to datetime format
df_btc = pd.read_csv('./data/BTCUSD.csv')
df_btc['Date'] = pd.to_datetime(df_btc['Date'])

# Set 'Date' as the index
df_btc.set_index('Date', inplace=True)

# Sort the DataFrame by the index
df_btc.sort_index(inplace=True)
df_btc_last_4_years = df_btc.last('4Y')

# Plot the data
plt.figure(figsize=(14, 8))
plt.plot(df_btc_last_4_years['Close'], label='Close Price')
plt.title('BTC Close Price for the Last 4 Years')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.show()