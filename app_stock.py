import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Retrieve database credentials from Streamlit secrets
db_user = st.secrets['DB_USER']
db_password = st.secrets['DB_PASSWORD']
db_host = st.secrets['DB_HOST']
db_name = st.secrets['DB_NAME']
db_table = st.secrets['DB_TABLE']
db_port = st.secrets['DB_PORT']

# Establish a connection to PostgreSQL and load data
def load_data():
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    query = f'SELECT * FROM {db_table}'
    data = pd.read_sql(query, engine)
    return data

data = load_data()

# Function to calculate daily percentage change
def calculate_percentage_change(df):
    df['percentage_change'] = df['close'].pct_change() * 100
    return df

# Calculate daily percentage change for each stock
stocks = data['ticker'].unique()
percentage_changes = []

for stock in stocks:
    stock_df = data[data['ticker'] == stock].copy()
    stock_df = calculate_percentage_change(stock_df)
    percentage_changes.append(stock_df)

# Concatenate the updated dataframes
df = pd.concat(percentage_changes)

# Streamlit app
st.title("Stock Market Dashboard")

# Display the latest average daily data for each stock
st.header("Latest Average Daily Data")
latest_data = df.groupby('ticker').tail(1)
latest_data.set_index('ticker', inplace=True)
st.dataframe(latest_data[['date', 'open', 'high', 'low', 'close', 'volume']])

# Display the daily percentage change for each stock
st.header("Daily Percentage Change")
percentage_change = latest_data[['date', 'percentage_change']]
st.dataframe(percentage_change)

# Plot volume traded bar chart
st.header("Volume Traded for Selected Stock")
selected_stock = st.selectbox('Select a stock', stocks)
stock_df = df[df['ticker'] == selected_stock]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(stock_df['date'], stock_df['volume'], color='skyblue', edgecolor='blue', label=selected_stock)

# Enhancements
ax.set_title(f'{selected_stock} Trading Volume', fontsize=20, fontweight='bold')
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Volume', fontsize=14, fontweight='bold')

ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
# Show the plot
plt.tight_layout()
st.pyplot(fig)

# Line graph for all stocks' closing prices
st.header("Weekly Closing Prices of All Stocks")
fig, ax = plt.subplots(figsize=(10, 5))

# Use seaborn color palette for better visuals
palette = sns.color_palette("tab10", len(stocks))
for i, stock in enumerate(stocks):
    stock_df = df[df['ticker'] == stock]
    ax.plot(stock_df['date'], stock_df['close'], label=stock, color=palette[i])

ax.set_title('Closing Prices of All Stocks')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.legend()
ax.grid(True)

# Set major ticks every week and format the date
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Make every 2nd tick label visible to avoid clutter
for i, label in enumerate(ax.get_xticklabels()):
    if i % 2 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)

# Rotate date labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

st.pyplot(fig)
