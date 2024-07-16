import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="Stock Market Dashboard", layout="centered")

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

# Check if 'ticker' column exists
if 'ticker' not in data.columns:
    st.error("The column 'ticker' does not exist in the data.")
else:
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

    st.write("")
    st.write("")

    # Display the latest average daily data for each stock
    st.header("Latest Average Daily Data")
    latest_data = df.groupby('ticker').tail(1)
    latest_data.set_index('ticker', inplace=True)

    # Drop the 'id' column if it exists
    columns_to_drop = ['id']
    latest_data = latest_data.drop(columns=[col for col in columns_to_drop if col in latest_data.columns])
    
    # Apply custom styling to the latest data table
    def style_specific_columns(row):
        color = 'lightgreen' if row['percentage_change'] > 0 else 'lightcoral'
        return ['background-color: lightblue' if col == 'ticker' else
                'background-color: lightgreen' if col == 'open' else
                'background-color: lightpink' if col == 'high' else
                'background-color: lightyellow' if col == 'date' else
                'background-color: lightgray' if col == 'close' else
                'background-color: lightblue' if col == 'volume' else
                f'background-color: {color}' if col == 'percentage_change' else
                '' for col in latest_data.columns]

    # Format the columns to 2 decimal places
    latest_data_style = latest_data.style.format({
        'open': '{:.2f}',
        'high': '{:.2f}',
        'low': '{:.2f}',
        'close': '{:.2f}',
        'volume': '{:.0f}',
        'percentage_change': '{:.4f}'
    }).apply(style_specific_columns, axis=1)
    st.dataframe(latest_data_style, width=900, height=213)

    # Plot volume traded bar chart
    st.header("Volume Traded for Selected Stock")
    selected_stock = st.selectbox('Select a stock', latest_data.index)
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
    palette = sns.color_palette("tab10", len(latest_data.index))
    for i, stock in enumerate(latest_data.index):
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
