import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Sample financial data
data = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'Stock Price': np.random.rand(100) * 100,
    'Portfolio Value': np.cumsum(np.random.randn(100) * 1000),
    'Returns': np.random.randn(100) / 100,
    'Volume': np.random.randint(1000, 10000, 100),
    'Interest Rate': np.random.rand(100) * 5,
})

# Create a Streamlit app
st.title("Finance Dashboard")

# Create a two-column layout
col1, col2,col3= st.columns(3)

# First row
with col1:
    st.header("Stock Price")
    fig1 = px.line(data, x='Date', y='Stock Price', title='Stock Price Over Time')
    st.plotly_chart(fig1, use_container_width=True)

with col1:
    st.header("Portfolio Value")
    fig2 = px.line(data, x='Date', y='Portfolio Value', title='Portfolio Value Over Time')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.header("Returns")
    fig3 = px.histogram(data, x='Returns', title='Returns Distribution')
    st.plotly_chart(fig3, use_container_width=True)

# Second row
with col2:
    st.header("Trading Volume")
    fig4 = px.bar(data, x='Date', y='Volume', title='Trading Volume Over Time')
    st.plotly_chart(fig4, use_container_width=True)

with col3:
    st.header("Interest Rate")
    fig5 = px.line(data, x='Date', y='Interest Rate', title='Interest Rate Over Time')
    st.plotly_chart(fig5, use_container_width=True)

with col3:
    st.header("Correlation Matrix")
    correlation_matrix = data.corr()
    fig6 = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                     x=correlation_matrix.index,
                                     y=correlation_matrix.columns,
                                     colorscale='Viridis'))
    st.plotly_chart(fig6, use_container_width=False)
