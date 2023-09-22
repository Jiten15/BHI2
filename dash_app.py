import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
#from query import *
import time
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.offline as pyo
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
# from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np


st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
st.subheader("Dashboard")
st.markdown("##")

theme_plotly = None # None or streamlit

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


df=pd.read_excel('ad.xlsx')


df['year'] = pd.to_datetime(df['Invoice Date']).dt.year
df['month'] = pd.to_datetime(df['Invoice Date']).dt.month
df['day'] = pd.to_datetime(df['Invoice Date']).dt.day



def feature_predictions():

	def cat_var(a):
		return pd.factorize(a)[0]

	df1=df.copy()

	options_Retailer = ['Foot Locker','Walmart','Sports Direct','West Gear',"Kohl's",'Amazon']
	options_Region = ['Northeast', 'South', 'West', 'Midwest', 'Southeast']
	options_State = ['New York', 'Texas', 'California', 'Illinois', 'Pennsylvania','Nevada', 'Colorado', 'Washington', 'Florida', 'Minnesota','Montana',
	'Tennessee', 'Nebraska', 'Alabama', 'Maine', 'Alaska','Hawaii', 'Wyoming', 'Virginia', 'Michigan', 'Missouri', 'Utah','Oregon', 'Louisiana', 'Idaho',
	'Arizona', 'New Mexico', 'Georgia','South Carolina', 'North Carolina', 'Ohio', 'Kentucky','Mississippi', 'Arkansas', 'Oklahoma', 'Kansas', 'South Dakota',
	'North Dakota', 'Iowa', 'Wisconsin', 'Indiana', 'West Virginia','Maryland', 'Delaware', 'New Jersey', 'Connecticut','Rhode Island',
	'Massachusetts', 'Vermont', 'New Hampshire']
	options_City = ['New York', 'Houston', 'San Francisco', 'Los Angeles', 'Chicago','Dallas', 'Philadelphia', 'Las Vegas', 'Denver', 'Seattle',
	'Miami', 'Minneapolis', 'Billings', 'Knoxville', 'Omaha','Birmingham', 'Portland', 'Anchorage', 'Honolulu', 'Orlando',
	'Albany', 'Cheyenne', 'Richmond', 'Detroit', 'St. Louis','Salt Lake City', 'New Orleans', 'Boise', 'Phoenix', 'Albuquerque',
	'Atlanta', 'Charleston', 'Charlotte', 'Columbus', 'Louisville','Jackson', 'Little Rock', 'Oklahoma City', 'Wichita','Sioux Falls', 'Fargo',
	'Des Moines', 'Milwaukee', 'Indianapolis','Baltimore', 'Wilmington', 'Newark', 'Hartford', 'Providence','Boston', 'Burlington', 'Manchester']
	options_Product = ["Men's Street Footwear", "Men's Athletic Footwear","Women's Street Footwear", "Women's Athletic Footwear",
	"Men's Apparel", "Women's Apparel"]
	options_Method = ['In-store', 'Outlet', 'Online']



	# Display a selectbox widget to allow the user to choose an option
	feature1 = st.selectbox("Retailer", options_Retailer)
	df1.loc[df.index[-1] + 1,"Retailer"] = feature1

	feature2 = st.selectbox("Region", options_Region)
	df1.loc[df.index[-1] + 1,"Region"] = feature2

	feature3 = st.selectbox("State", options_State)
	df1.loc[df.index[-1] + 1,"State"] = feature3

	feature4 = st.selectbox("City", options_City)
	df1.loc[df.index[-1] + 1,"City"] = feature4

	feature5 = st.selectbox("Product", options_Product)
	df1.loc[df.index[-1] + 1,"Product"] = feature5

	feature6 = st.sidebar.number_input("Price per Unit", value=50)
	df1.loc[df.index[-1] + 1,"Price per Unit"] = feature6	

	feature7 = st.sidebar.number_input("Units Sold", value=1300)
	df1.loc[df.index[-1] + 1,"Units Sold"] = feature7

	feature8 = st.sidebar.number_input("Operating Profit", value=300000)
	df1.loc[df.index[-1] + 1,"Operating Profit"] = feature8

	feature9 = st.sidebar.number_input("Operating Margin(%)", value=30)
	df1.loc[df.index[-1] + 1,"Operating Margin"] = feature9

	feature10 = st.selectbox("Sales Method", options_Method)
	df1.loc[df.index[-1] + 1,"Sales Method"] = feature10

	df2=df1.copy()

	df2['Retailer']=pd.factorize(df2.Retailer)[0]
	df2['Region']=pd.factorize(df2.Region)[0]
	df2['State']=pd.factorize(df2.State)[0]
	df2['City']=pd.factorize(df2.City)[0]
	df2['Product']=pd.factorize(df2.Product)[0]
	

	df2.rename(columns={'Sales Method':'Method'},inplace=True)

	df2['Method']=pd.factorize(df2.Method)[0]

	df2 = df2.drop('Retailer ID',axis=1)
	df2 = df2.drop('Invoice Date',axis=1)

	df2['Units Sold'] = df2['Units Sold'].astype(int)
	df2['Total Sales']=df2['Total Sales'].astype(float)
	df2['Operating Profit'] = df2['Operating Profit'].astype(float)
	

	X= df2.values[:,(0,1,2,3,4,5,6,8,9,10)]
	Y= df2.values[:,7]

	# X_test=np.array([X[-1]])

	X_t=X[:-2]
	y_t=Y[:-2]

	X_train, _, y_train, _  = train_test_split(X_t, y_t, test_size = 0.25, random_state = 42)

	lr= LinearRegression()

	lr.fit(X_train,y_train)


	# X_test = np.array([[cat_var(feature1), feature2,feature3, feature4,feature5, feature6,feature7, feature8,feature9,feature10]])
	if st.button("Predict Total Sales"):
		y_pred = lr.predict([np.array(X[-1])])
		st.title(f"The Total Sales will be : {round(abs(y_pred[0]),2)}$")
		# st.write(round(abs(y_pred[0]),2))





def Home():
	col1, col2,col3= st.columns(3)
	with col1:
	    ####### Retailer Counts Bar Chart #########
		retailer_counts = df['Retailer'].value_counts().reset_index()
		retailer_counts.columns = ['Retailer', 'Count']
		retailer_counts = retailer_counts.sort_values(by='Count', ascending=False)
		# st.title("Retailer Counts Bar Chart")
		fig1 = px.bar(retailer_counts, x='Retailer', y='Count', title='Retailer Counts')
		st.plotly_chart(fig1,use_container_width=True)

	with col1:
	    # Group the data by retailer and sum the total sales for each retailer
		retailer_sales = df.groupby('Retailer')['Total Sales'].sum().reset_index()
		# Calculate the total sales of all retailers
		total_sales = retailer_sales['Total Sales'].sum()
		# Calculate the market share of each retailer by dividing their total sales by the total sales of all retailers
		retailer_sales['Market Share'] = retailer_sales['Total Sales'] / total_sales
		# st.title("Market Share Pie Chart")
		fig2 = px.pie(retailer_sales, values='Market Share', names='Retailer', title='Market Share of Retailers')
		st.plotly_chart(fig2,use_container_width=True)

	with col2:
	   product_sales = df.groupby(['Retailer', 'Product'])['Total Sales'].sum().reset_index()
	   fig3 = px.bar(product_sales,x='Retailer',y='Total Sales',color='Product',barmode ='group',title='Total Sales by Product and Retailer',labels={'Total Sales':'Sales'},
			category_orders={"Retailer":["Retailer A","Retailer B","Retailer C"]})
	   st.plotly_chart(fig3,use_container_width=True)

	# Second row
	with col2:
	    # Convert the 'year' column to a datetime format
		df['year'] = pd.to_datetime(df['year'], format='%Y')
		# Group the data by Region and year and calculate the total Sales for each group
		region_sales = df.groupby(['Region', 'year'])['Total Sales'].sum().reset_index()
		# st.title("Total Sales by Region Over Time")
		# Create a line chart using Plotly Express
		fig4 = px.line(region_sales, x='year', y='Total Sales', color='Region',
		              title='Total Sales by Region Over Time')
		# Customize the date formatting on the x-axis
		fig4.update_xaxes(
		    dtick="M1",  # Sets the tick frequency to monthly
		    tickformat="%b-%Y"  # Formats the date as "mm-yyyy"
		)
		# Display the line chart in the Streamlit app
		st.plotly_chart(fig4,use_container_width=True)

	with col3:
	    # Group the data by Sales Method and calculate the total Operating Profit for each group
		profit_by_method = df.groupby('Sales Method')['Operating Profit'].sum().reset_index()
		# st.title("Total Operating Profit by Sales Method")
		# Create a bar chart using Plotly Express
		fig5 = px.bar(profit_by_method, x='Sales Method', y='Operating Profit',title='Total Operating Profit by Sales Method')
		# Display the bar chart in the Streamlit app
		st.plotly_chart(fig5,use_container_width=True)

	with col3:
	    # Group the data by Sales Method and calculate the average Total Sales for each group
		sales_by_method = df.groupby('Sales Method')['Total Sales'].mean().reset_index()
		# st.title("Average Total Sales by Sales Method")
		fig6 = px.bar(sales_by_method, x='Sales Method', y='Total Sales',title='Average Total Sales by Sales Method')
		st.plotly_chart(fig6,use_container_width=True)


def sideBar():

 with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Home","Predictions","Plot a Feature","Compare Two Durations of a Feature","Forecast a Feature"],
        icons=["house"],
        menu_icon="cast",
        default_index=0
    )
 if selected=="Home":
    #st.subheader(f"Page: {selected}")
    Home()
    # graphs()
 if selected=="Plot a Feature":
    st.subheader(f"Page: {selected}")
    feature_1()
    # Progressbar()
    # graphs()

 if selected=="Compare Two Durations of a Feature":
    st.subheader(f"Page: {selected}")
    feature_2()      

 if selected=="Forecast a Feature":
    st.subheader(f"Page: {selected}")
    feature_3()

 if selected=="Predictions":
    st.subheader(f"Page: {selected}")
    feature_predictions()

sideBar()

# Home()
# feature_1()
# feature_2()
# feature_3()


#theme
hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""

    