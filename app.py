import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px


df=pd.read_excel('ad.xlsx')


df['year'] = pd.to_datetime(df['Invoice Date']).dt.year
df['month'] = pd.to_datetime(df['Invoice Date']).dt.month
df['day'] = pd.to_datetime(df['Invoice Date']).dt.day


####### Retailer Counts Bar Chart #########
retailer_counts = df['Retailer'].value_counts().reset_index()
retailer_counts.columns = ['Retailer', 'Count']

# Sort the retailer counts in descending order
retailer_counts = retailer_counts.sort_values(by='Count', ascending=False)

# Streamlit app
st.title("Retailer Counts Bar Chart")

# Create a bar chart using Plotly Express
fig = px.bar(retailer_counts, x='Retailer', y='Count', title='Retailer Counts')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig)


############ Market Share Pie Chart #############
# Group the data by retailer and sum the total sales for each retailer
retailer_sales = df.groupby('Retailer')['Total Sales'].sum().reset_index()

# Calculate the total sales of all retailers
total_sales = retailer_sales['Total Sales'].sum()

# Calculate the market share of each retailer by dividing their total sales by the total sales of all retailers
retailer_sales['Market Share'] = retailer_sales['Total Sales'] / total_sales

# Streamlit app
st.title("Market Share Pie Chart")

# Create a pie chart using Plotly Express
fig = px.pie(retailer_sales, values='Market Share', names='Retailer', title='Market Share of Retailers')

# Display the pie chart in the Streamlit app
st.plotly_chart(fig)


######### Total Sales by Product and Retailer ##########

product_sales = df.groupby(['Retailer', 'Product'])['Total Sales'].sum().reset_index()
# product_sales_df = product_sales.unstack(level=0)
# Create a grouped bar chart using Plotly Express
fig = px.bar(product_sales, x='Retailer', y='Total Sales', color='Product',barmode = 'group',
             title='Total Sales by Product and Retailer',
             labels={'Total Sales': 'Sales'},
             category_orders={"Retailer": ["Retailer A", "Retailer B", "Retailer C"]})

# Show the Plotly chart
st.plotly_chart(fig)


############################
# Group the data by Sales Method and calculate the average Total Sales for each group
sales_by_method = df.groupby('Sales Method')['Total Sales'].mean().reset_index()

# Streamlit app
st.title("Average Total Sales by Sales Method")

# Create a bar chart using Plotly Express
fig = px.bar(sales_by_method, x='Sales Method', y='Total Sales',
             title='Average Total Sales by Sales Method')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig)


############################
# Group the data by Sales Method and calculate the total Operating Profit for each group
profit_by_method = df.groupby('Sales Method')['Operating Profit'].sum().reset_index()

# Streamlit app
st.title("Total Operating Profit by Sales Method")

# Create a bar chart using Plotly Express
fig = px.bar(profit_by_method, x='Sales Method', y='Operating Profit',
             title='Total Operating Profit by Sales Method')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig)

##############################
# Convert the 'year' column to a datetime format
df['year'] = pd.to_datetime(df['year'], format='%Y')

# Group the data by Region and year and calculate the total Sales for each group
region_sales = df.groupby(['Region', 'year'])['Total Sales'].sum().reset_index()

# Streamlit app
st.title("Total Sales by Region Over Time")

# Create a line chart using Plotly Express
fig = px.line(region_sales, x='year', y='Total Sales', color='Region',
              title='Total Sales by Region Over Time')

# Customize the date formatting on the x-axis
fig.update_xaxes(
    dtick="M1",  # Sets the tick frequency to monthly
    tickformat="%b-%Y"  # Formats the date as "mm-yyyy"
)

# Display the line chart in the Streamlit app
st.plotly_chart(fig)