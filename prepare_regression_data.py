import pandas as pd

df = pd.read_csv('cleaned_product_grouping.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

agg = df.groupby(['StockCode', 'Year', 'Month']).agg({
    'NetRevenue': 'sum',
    'NetQuantity': 'sum',
    'CustomerFrequency': 'mean',
    'ProductFrequency': 'mean'
}).reset_index()

agg['NetRevenue_LastMonth'] = agg.groupby('StockCode')['NetRevenue'].shift(1)
agg['NetRevenue_MA3'] = agg.groupby('StockCode')['NetRevenue'].rolling(3).mean().shift(1).reset_index(0, drop=True)
agg['NextMonthRevenue'] = agg.groupby('StockCode')['NetRevenue'].shift(-1)
agg = agg.dropna()

agg.to_csv('dataset/processed/product_revenue_dataset.csv', index=False)
print("✅ Aggregated dataset saved as 'dataset/processed/product_revenue_dataset.csv'")