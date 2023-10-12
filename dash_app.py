import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
#from query import *
import time

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
from datetime import date

from dateutil import parser
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
from surprise import Dataset, Reader, SVD

import transformers
import spacy
import re


import streamlit.components.v1 as components

import speech_recognition as sr
from datetime import datetime, timedelta
import calendar





st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
st.subheader("Dashboard")
st.markdown("##")

theme_plotly = None # None or streamlit

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


df=pd.read_excel('ad.xlsx')

# 'Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City',
#        'Product', 'Price per Unit', 'Units Sold', 'Total Sales',
#        'Operating Profit', 'Operating Margin', 'Sales Method', 'Date'


df['year'] = pd.to_datetime(df['Invoice Date']).dt.year
df['month'] = pd.to_datetime(df['Invoice Date']).dt.month
df['day'] = pd.to_datetime(df['Invoice Date']).dt.day


def assistant():

	# Create a Streamlit app
	st.title("Chatbot")

	# Create a function to convert speech to text
	def speech_to_text():
	    recognizer = sr.Recognizer()
	    microphone = sr.Microphone()

	    with microphone as source:
	        st.write("Please speak...")
	        recognizer.adjust_for_ambient_noise(source)
	        audio = recognizer.listen(source)
	        st.write("Recording complete.")

	    try:
	        text = recognizer.recognize_google(audio)
	        st.success("Text: " + text)
	    except sr.UnknownValueError:
	        st.error("Could not understand the audio.")
	    except sr.RequestError as e:
	        st.error("Could not request results; {0}".format(e))
	    return str(text)    

	

	# Load spaCy for natural language understanding
	nlp = spacy.load("en_core_web_sm")

	# Define regular expressions for matching different components
	column_pattern = r'(total sales|price per unit|units sold|operating profit|operating margin)'
	date_pattern = r'\d{4}-\d{2}-\d{2}'
	time_period_pattern = r'(monthly|quarterly|yearly|daily)'

	def extract_info(user_message):
	    # Initialize variables with default values
	    selected_columns = None
	    start_date = None
	    end_date = None
	    time_period = None

	    # Convert the user's input to lowercase to make it case-insensitive
	    user_message = user_message.lower()

	    # Find all occurrences of selected columns
	    selected_columns = re.findall(column_pattern, user_message)

	    # Find start date and end date
	    dates = re.findall(date_pattern, user_message)
	    if len(dates) >= 2:
	        start_date = datetime.strptime(dates[0], '%Y-%m-%d').date()
	        end_date = datetime.strptime(dates[1], '%Y-%m-%d').date()

	    # Find all occurrences of time period
	    time_periods = re.findall(time_period_pattern, user_message)

	    # Check if selected_columns is a list, and join them into a single string
	    if selected_columns:
	        selected_columns = ' '.join(selected_columns)

	    # Check if time_periods is a list, and join them into a single string
	    if time_periods:
	        time_period = ' '.join(time_periods)

	    return selected_columns, start_date, end_date, time_period

	user_id_pattern = r'user (\d{3})'
	no_of_recom_pattern = r'list of (\d+) product recommendations'

	def user_id_extract_info(user_message):
	    user_id = None
	    no_of_recom = None

	    # Convert the user's input to lowercase to make it case-insensitive
	    user_message = user_message.lower()

	    # Find user_id values
	    user_id_values = re.findall(user_id_pattern, user_message)
	    if user_id_values:
	        user_id = int(user_id_values[0])

	    # Find no_of_recom values
	    no_of_recom_values = re.findall(no_of_recom_pattern, user_message)
	    if no_of_recom_values:
	        no_of_recom = int(no_of_recom_values[0])

	    return user_id, no_of_recom

	def date_extract(text):
		parsed_dates = parser.parse(text, fuzzy=True, dayfirst=True)
		parsed_dates = [date for date in parsed_dates if date.year]
		if len(parsed_dates) >= 2:
			parsed_dates.sort()
			start_date = parsed_dates[0].strftime('%Y-%m-%d')
			end_date = parsed_dates[-1].strftime('%Y-%m-%d')
		return start_date,end_date


	def date_extraction_from_audio(input_text):
	    month_mapping = {
	    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
	    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
	    }

	    # Regular expression to match month-year pattern
	    pattern = r'(\w{3,}) (\d{4})'

	    # Search for month-year matches in the input text
	    matches = re.findall(pattern, input_text.lower())

	    # Function to get the date as a datetime object from the matched month-year pair
	    def get_date(month, year):
	        month_number = month_mapping.get(month[:3], 1)  # Default to January if month is not found
	        return datetime(int(year), month_number, 1)

	    # Initialize start_date and end_date
	    start_date = end_date = None

	    # Process the matches to extract start_date and end_date
	    if len(matches) >= 2:
	        date1 = get_date(matches[0][0], matches[0][1])
	        date2 = get_date(matches[1][0], matches[1][1])
	        
	        # Compare the dates and determine start_date and end_date
	        if date1 < date2:
	            start_date = date1
	            end_date = date2
	        else:
	            start_date = date2
	            end_date = date1

	    # Convert end_date to the last day of the month
	    if end_date:
	        year, month, last_day = end_date.year, end_date.month, calendar.monthrange(end_date.year, end_date.month)[1]
	        end_date = datetime(year, month, last_day)

	    return start_date,end_date


	def extract_growth_rate(input_text):
	    pattern = r'(\d+(?:\.\d+)?)\s*%'
	    match = re.search(pattern, input_text)
	    growth_rate = None
	    if match:
	        growth_rate = float(match.group(1))

	    return growth_rate


	# Chat interface
	# Create a button to trigger speech recognition

	st.write("Chat with the bot:")
	user_message = st.text_input("You: ")

	if user_message:
		# Process user input with spaCy for intent recognition
		doc = nlp(user_message)
		# st.write(doc)
		intent = None

		# Recognize intents related to different sections
		for token in doc:
		  if token.dep_ == "dative" and token.head.lemma_ == "call":
		      intent = "call_function"
		      break
		  elif "plot" in user_message.lower():
		      intent = "plot"
		      break
		  elif "compare" in user_message.lower() and "duration" in user_message.lower():
		      intent = "compare_duration"
		      break
		  elif "forecast" in user_message.lower():
		      intent = "forecast"
		      break
		  elif "growth rate" in user_message.lower():
		      intent = "growth rate"
		      break
		  elif "recommendations" in user_message.lower():
		      intent = "recommendations"
		      break

		# Generate a chatbot response based on the recognized intent
		if intent == "call_function":
			st.write("Calling a function... (replace this with your function call code)")
			response = "Function called successfully."

		# Plot
		elif intent == "plot":
			# response = "Sure, let's go to the Plotting section."
			selected_column, start_date, end_date, time_period = extract_info(user_message)
			selected_column=selected_column.title()

			# st.write(f"{selected_column}, {start_date}, {end_date}, {time_period}")
			if start_date is None :

				start_date, end_date = date_extraction_from_audio(user_message)

				if start_date is None:
					st.write("sorry! could not get the date. Would you please select it from below options")
					start_date = st.date_input("Select a Start Date")
					end_date = st.date_input("Select an End Date")
				
				

			if start_date and end_date is not None:

				df_new = df.groupby('Invoice Date').agg({
				'Price per Unit': 'sum',
			 	'Units Sold': 'sum',
			 	'Total Sales': 'sum',
			 	'Operating Profit': 'sum',
			 	'Operating Margin': 'mean'}).reset_index()
				df_new['Price Per Unit']=df_new['Price per Unit']


				df_new['Invoice Date'] = pd.to_datetime(df_new['Invoice Date'])


				start_date = datetime.combine(start_date, datetime.min.time())
				end_date = datetime.combine(end_date, datetime.max.time())

				if time_period == "monthly":
					dates = pd.date_range(start=start_date,end=end_date,freq="MS")

				elif time_period == "quarterly":
					dates = pd.date_range(start=start_date,end=end_date,freq="QS")
				elif time_period == "yearly":
					dates = pd.date_range(start=start_date,end=end_date,freq="Y")
				elif time_period == "daily":
					dates = pd.date_range(start=start_date,end=end_date,freq="D")

				filtered_df=df_new[df_new['Invoice Date'].isin(dates)]

				def plot(x1,y1):
					trace1 = go.Scatter(x=x1,y=y1, mode='lines+markers', name=f'{time_period.title()}')
					layout = go.Layout(title=f"{selected_column}")
					fig = go.Figure(data=[trace1], layout=layout)
					st.plotly_chart(fig)

				plot(filtered_df['Invoice Date'],filtered_df[selected_column])

			else:
				st.write("Please try again. Not able to get you this time.")


		# Compare Plots
		elif intent == "compare_duration":
			response = "Okay, let's compare durations."
		
		# Forecast
		elif intent == "forecast":
			# response = "Great, let's do some forecasting."
			selected_column, start_date, end_date, frequency = extract_info(user_message)

			if start_date is None :

				start_date, end_date = date_extraction_from_audio(user_message)

				if start_date is None:
					st.write("sorry! could not get the date. Would you please select it from below options Or try again")
					start_date = st.date_input("Select a Start Date")
					end_date = st.date_input("Select an End Date")
				
				

			if start_date != end_date:

				df['Date'] = pd.to_datetime(df['Invoice Date'])
				selected_column=selected_column.title()

				df2= df[['Date','Total Sales','Price per Unit','Units Sold','Operating Profit','Operating Margin']]
				df2['Price Per Unit']=df2['Price per Unit']
				df2 = df2.groupby('Date')[[selected_column]].sum().reset_index()

				df2.set_index('Date', inplace=True)

				model=sm.tsa.statespace.SARIMAX(df2[selected_column],order=(1, 1, 1),seasonal_order=(1,1,1,12))
				results=model.fit()
				

				if frequency=="monthly":
				    frequency='MS'
				elif frequency=="quarterly":
				    frequency='QS'
				elif frequency=="yearly":
				    frequency='Y'
				elif frequency=="daily":
				    frequency='D'    
				    
				index_future_dates=pd.date_range(start=start_date,end=end_date,freq=frequency)
				#print(index_future_dates
				C=len(index_future_dates)
				pred=results.predict(start=len(df2),end=len(df2)+C-1,typ='levels').rename('ARIMA Predictions')
				#print(comp_pred)
				pred.index=index_future_dates
				# print(pred)

				fig = go.Figure()

				fig.add_trace(go.Scatter(x=pred.index, y=pred, mode='lines', name='Forecast Sales'))

				# Update the layout of the plot
				fig.update_layout(title=f'Forecasted {selected_column}',
				                  xaxis_title='Date',
				                  yaxis_title=f'{selected_column}')

				# Show the plot
				st.plotly_chart(fig)


		# Budget Planning
		elif intent == "growth rate":

				df['Date'] = pd.to_datetime(df['Invoice Date'])
				df2 = df[['Date', 'Total Sales']]
				df2 = df2.groupby('Date')[['Total Sales']].sum().reset_index()
				df2.set_index('Date', inplace=True)
				model = sm.tsa.statespace.SARIMAX(df2['Total Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
				results = model.fit()
				frequency = 'D'
				index_future_dates = pd.date_range(start="2020-01-01", end="2021-04-02", freq=frequency)
				C = len(index_future_dates)
				pred = results.predict(start=len(df2), end=len(df2) + C - 1, typ='levels').rename('ARIMA Predictions')
				pred.index = index_future_dates
				x = pred.index
				y = pred
				data1 = pd.DataFrame({'Date': x, 'Forecast': y})
				data1['Date'] = pd.to_datetime(data1['Date'])
				last_date = "2021-04-02"
				last_quarter_start = pd.Timestamp(year=2021, month=1 + 1, day=1)
				last_quarter_data = data1[(data1['Date'] >= last_quarter_start) & (data1['Date'] <= last_date)]
				total_last_quarter_forecast = np.sum(last_quarter_data['Forecast'])
				initial_company_budget = 90000000
				i1=initial_company_budget

				#input 1
				growth_rate = extract_growth_rate(user_message)
				st.write(growth_rate)


				quarterly_sales = df2.resample('Q')['Total Sales'].sum()
				last_quarter_sales = quarterly_sales.iloc[-1]
				revenues = last_quarter_sales  # Example initial revenue
				last_budget_forecast = total_last_quarter_forecast  # Example last budget forecast
				company_expenses = {}
				expense_categories = ['Salaries', 'Utilities', 'Rent', 'Marketing', 'Supplies', 'Other']
				company_expenses['Salaries'] = initial_company_budget*0.4
				company_expenses['Utilities'] = initial_company_budget*0.1
				company_expenses['Rent'] = initial_company_budget*0.15
				company_expenses['Marketing'] = initial_company_budget*0.2
				company_expenses['Supplies'] = initial_company_budget*0.1
				company_expenses['Other'] = initial_company_budget*0.05

				data = {
			    'Category': expense_categories,
			    'Expense': list(company_expenses.values())
				}

				df11 = pd.DataFrame(data)
				fig = px.bar(df11, x='Category', y='Expense', title='Company Expenses')
				# st.title('Company Expenses')
				# st.plotly_chart(fig, use_container_width=True)

				#############
				initial_product_budget = {}
				products = ['Product1', 'Product2', 'Product3']
				initial_product_budget['Product1']=0.15*initial_company_budget
				initial_product_budget['Product2']=0.6*initial_company_budget
				initial_product_budget['Product3']=0.25*initial_company_budget
				new_product_budget = {}
				products = ['Product1', 'Product2', 'Product3']
				for product in products:
				    budget = float(initial_product_budget[str(product)]*growth_rate*0.01)
				    if budget:
				        new_product_budget[product] = budget
				data13 = {
			    'Products': list(initial_product_budget.keys()) +list(new_product_budget.keys()),
			    'Allocations': list(initial_product_budget.values()) + list(new_product_budget.values()),
			    'BudgetType': ['Initial'] * len(initial_product_budget) + ['New'] * len(new_product_budget)
				}
				df13 = pd.DataFrame(data13)
				fig13 = px.bar(df13, x='Products', y='Allocations', color='BudgetType',title='Product Budget Allocation')
				st.title('Product Budget Allocation')
				st.plotly_chart(fig13, use_container_width=True) 
				company_expenses = sum(company_expenses.values())
				product_expenses = {product: 0 for product in initial_product_budget}

				# Function to check if spending is as per the budget
				def is_spending_within_budget(company_budget, product_budgets, company_expenses, product_expenses):
				    total_company_expenses = sum(product_expenses.values()) + company_expenses
				    return total_company_expenses <= company_budget

				# Function to calculate and track variances from the last budget forecast
				def calculate_variance(revenues, last_forecast):
				    return revenues - last_forecast

				# Check if spending is within the company and product budgets
				if is_spending_within_budget(initial_company_budget, initial_product_budget, company_expenses, product_expenses):
				    st.write("Spending is within the budget.")
				else:
				    st.write("Spending exceeds the budget.")

				# Calculate and track variances
				variance = calculate_variance(revenues, last_budget_forecast)
				if variance < 0:
				    st.write("Revenues are showing a dip. Budget may need adjustment.")
				else:
				    st.write("Budget holds good even with current revenues.")

				# Function to allocate new budgets based on current revenues
				def allocate_budget_based_on_revenues(initial_budget, current_revenues):
				   return float(initial_budget+abs(initial_budget-current_revenues)* growth_rate*0.01)  # Adjust budget based on a percentage (e.g., 10%)

				def allocate_product_budget_based_on_revenues(budget, revenues):
					return float(budget+abs(budget-revenues)* growth_rate*0.01)

				new_company_budget = allocate_budget_based_on_revenues(initial_company_budget, revenues)
				new_product_budgets = {product: allocate_product_budget_based_on_revenues(budget, revenues) for product, budget in initial_product_budget.items()}

				data12 = {
			    'Category': ['Recent| Company Budget','New| Company Budget Allocation'],
			    'Allocations': [initial_company_budget, new_company_budget]
				}

				df12 = pd.DataFrame(data12)
				fig12 = px.bar(df12, x='Category', y='Allocations', title='Budget Allocation')
				st.title('Budget Allocation')
				st.plotly_chart(fig12, use_container_width=True)    

				################################################

				
				######################
				


		################ recommendations | bot ##########################
		elif intent == "recommendations":
			# response = "Certainly, let's explore recommendations."
			# Sample sales data
			user_id, no_of_recom=user_id_extract_info(user_message)
			# st.write(f"{user_id},{no_of_recom}")

			data = {
			    'user_id': [101, 102, 103, 104, 105, 106, 107, 108, 109],
			    'product_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009],
			    'rating': [5, 4, 2, 3, 1, 5, 4, 5, 3]
			}

			# Create a Surprise Reader object specifying the rating scale
			reader = Reader(rating_scale=(1, 5))  # Assuming ratings range from 1 to 5

			# Load the data into a Surprise Dataset
			data_surprise = Dataset.load_from_df(pd.DataFrame(data), reader)

			# Split the data into training and testing sets
			trainset = data_surprise.build_full_trainset()

			# Train the SVD (Singular Value Decomposition) algorithm
			model = SVD()
			model.fit(trainset)


			# user_id = 101  # Replace with the actual user ID
			products_rated_by_user = [product_id for user, product_id, rating in zip(data['user_id'], data['product_id'], data['rating']) if user == user_id]
			products_to_exclude = set(products_rated_by_user)

			# Generate recommendations for the user
			recommendations = []
			for product_id in data['product_id']:  # Assuming product IDs range from 1 to 5
			    if product_id not in products_to_exclude:
			        predicted_rating = model.predict(user_id, product_id).est
			        recommendations.append((product_id, predicted_rating))

			# Sort the recommendations by predicted rating in descending order
			recommendations.sort(key=lambda x: x[1], reverse=True)

			# Get the top N recommended product IDs (e.g., top 3)
			top_n_recommendations = [product_id for product_id, _ in recommendations[:5]]

			# Add your code to navigate to the Recommendations section
			st.write(f"products recommendations for user id {user_id} are : {top_n_recommendations}")

		else:
			st.write("Sorry! not able to get you please try again.")

		

def recommendations():

	# Sample sales data
	data = {
	    'user_id': [101, 102, 103, 104, 105, 106, 107, 108, 109],
	    'product_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009],
	    'rating': [5, 4, 2, 3, 1, 5, 4, 5, 3]
	}

	# Create a Surprise Reader object specifying the rating scale
	reader = Reader(rating_scale=(1, 5))  # Assuming ratings range from 1 to 5

	# Load the data into a Surprise Dataset
	data_surprise = Dataset.load_from_df(pd.DataFrame(data), reader)

	# Split the data into training and testing sets
	trainset = data_surprise.build_full_trainset()

	# Train the SVD (Singular Value Decomposition) algorithm
	model = SVD()
	model.fit(trainset)

	# Recommend products for a specific user (replace 'user_id' with the actual user ID)

	# Ask the user to input their user_id
	user_id = st.text_input("Enter your User ID:", value="102")

	# Display the user_id
	st.write(f"You entered User ID: {user_id}")

	# user_id = 101  # Replace with the actual user ID
	products_rated_by_user = [product_id for user, product_id, rating in zip(data['user_id'], data['product_id'], data['rating']) if user == user_id]
	products_to_exclude = set(products_rated_by_user)

	# Generate recommendations for the user
	recommendations = []
	for product_id in data['product_id']:  # Assuming product IDs range from 1 to 5
	    if product_id not in products_to_exclude:
	        predicted_rating = model.predict(user_id, product_id).est
	        recommendations.append((product_id, predicted_rating))

	# Sort the recommendations by predicted rating in descending order
	recommendations.sort(key=lambda x: x[1], reverse=True)

	# Get the top N recommended product IDs (e.g., top 3)
	top_n_recommendations = [product_id for product_id, _ in recommendations[:4]]

	st.write(f"Top 4 Product Recommendations for User {user_id}:")
	st.write(top_n_recommendations)




def qqt():

	st.title("Budget Planning")

	df['Date'] = pd.to_datetime(df['Invoice Date'])
	df2 = df[['Date', 'Total Sales']]
	df2 = df2.groupby('Date')[['Total Sales']].sum().reset_index()
	df2.set_index('Date', inplace=True)

	model = sm.tsa.statespace.SARIMAX(df2['Total Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
	results = model.fit()

	frequency = 'D'
	

	index_future_dates = pd.date_range(start="2020-01-01", end="2021-04-02", freq=frequency)
	C = len(index_future_dates)
	pred = results.predict(start=len(df2), end=len(df2) + C - 1, typ='levels').rename('ARIMA Predictions')
	pred.index = index_future_dates

	x = pred.index
	y = pred

	# Create a DataFrame from your data
	data1 = pd.DataFrame({'Date': x, 'Forecast': y})

	# Convert the 'Date' column to a datetime data type
	data1['Date'] = pd.to_datetime(data1['Date'])

	# Determine the last date in the data
	last_date = "2021-04-02"

	# Calculate the first date of the last quarter
	last_quarter_start = pd.Timestamp(year=2021, month=1 + 1, day=1)

	# Filter the data for the last quarter
	last_quarter_data = data1[(data1['Date'] >= last_quarter_start) & (data1['Date'] <= last_date)]

	# st.write(last_quarter_data)


	# Calculate the total forecast value for the last quarter
	total_last_quarter_forecast = np.sum(last_quarter_data['Forecast'])

	# Print the total forecast value for the last quarter
	# st.write(f"Total Forecast Value for the Last Quarter: {total_last_quarter_forecast}")

	initial_company_budget = st.number_input("Enter the Initial Company Budget:",value=90000000, min_value=0)
	i1=initial_company_budget


	# Example growth rate 
	growth_rate = st.number_input("Enter the Growth Rate you want on recent revenue:",value=0.1, min_value=0.0)







	quarterly_sales = df2.resample('Q')['Total Sales'].sum()

	# Get the total sales for the last quarter
	last_quarter_sales = quarterly_sales.iloc[-1]	
	# st.write(last_quarter_sales)        

	revenues = last_quarter_sales  # Example initial revenue
	last_budget_forecast = total_last_quarter_forecast  # Example last budget forecast







	# Title for the Streamlit app
	# st.title("Company Expense Tracker")

	# Create an empty dictionary to store company expenses
	company_expenses = {}

	# Define expense categories and ask users to input expenses
	expense_categories = ['Salaries', 'Utilities', 'Rent', 'Marketing', 'Supplies', 'Other']

	# for category in expense_categories:
	#     expense_amount = st.number_input(f"Enter the expense for {category}:",value=1000000, key=category, min_value=0)
	#     if expense_amount:
	company_expenses['Salaries'] = initial_company_budget*0.4
	company_expenses['Utilities'] = initial_company_budget*0.1
	company_expenses['Rent'] = initial_company_budget*0.15
	company_expenses['Marketing'] = initial_company_budget*0.2
	company_expenses['Supplies'] = initial_company_budget*0.1
	company_expenses['Other'] = initial_company_budget*0.05

	data = {
    'Category': expense_categories,
    'Expense': list(company_expenses.values())
	}

	df11 = pd.DataFrame(data)

	# Create a bar graph using Plotly Express
	fig = px.bar(df11, x='Category', y='Expense', title='Company Expenses')

	# Streamlit app
	st.title('Company Expenses')
	st.plotly_chart(fig, use_container_width=True)

	#############


	# Title for the Streamlit app
	# st.title("Product Budget Calculator")

	# Create an empty dictionary to store the initial product budget
	initial_product_budget = {}

	# Define the product names
	products = ['Product1', 'Product2', 'Product3']

	# Create input fields for each product's budget
	# for product in products:
	#     budget = st.number_input(f"Enter the initial budget for {product}:", key=product, value=1800000,min_value=0)
	#     if budget:
	#         initial_product_budget[product] = budget

	initial_product_budget['Product1']=0.15*initial_company_budget
	initial_product_budget['Product2']=0.6*initial_company_budget
	initial_product_budget['Product3']=0.25*initial_company_budget

	# Display the entered product budgets
	# if initial_product_budget:
	#     st.write("Initial Product Budgets:")
	#     for product, budget in initial_product_budget.items():
	#         st.write(f"{product}: ${budget:.2f}")


	# Create an empty dictionary to store the New product budget
	new_product_budget = {}

	# Define the product names
	products = ['Product1', 'Product2', 'Product3']

	# Create input fields for each product's budget
	for product in products:
	    budget = initial_product_budget[str(product)]*growth_rate
	    if budget:
	        new_product_budget[product] = budget

	#initial_product_budget
	data13 = {
    'Products': list(initial_product_budget.keys()) +list(new_product_budget.keys()),
    'Allocations': list(initial_product_budget.values()) + list(new_product_budget.values()),
    'BudgetType': ['Initial'] * len(initial_product_budget) + ['New'] * len(new_product_budget)
	}

	df13 = pd.DataFrame(data13)

	# Create a bar graph using Plotly Express
	fig13 = px.bar(df13, x='Products', y='Allocations', color='BudgetType',title='Product Budget Allocation')

	# Streamlit app
	st.title('Product Budget Allocation')
	st.plotly_chart(fig13, use_container_width=True) 








	
	# # Display the entered company expenses
	# if company_expenses:
	#     st.write("Company Expenses:")
	#     for category, amount in company_expenses.items():
	#         st.write(f"{category}: ${amount:.2f}")

	company_expenses = sum(company_expenses.values())
	product_expenses = {product: 0 for product in initial_product_budget}

	# Function to check if spending is as per the budget
	def is_spending_within_budget(company_budget, product_budgets, company_expenses, product_expenses):
	    total_company_expenses = sum(product_expenses.values()) + company_expenses
	    return total_company_expenses <= company_budget

	# Function to calculate and track variances from the last budget forecast
	def calculate_variance(revenues, last_forecast):
	    return revenues - last_forecast

	# Check if spending is within the company and product budgets
	if is_spending_within_budget(initial_company_budget, initial_product_budget, company_expenses, product_expenses):
	    st.write("Spending is within the budget.")
	else:
	    st.write("Spending exceeds the budget.")

	# Calculate and track variances
	variance = calculate_variance(revenues, last_budget_forecast)





	# Check if the budget still holds good if revenues are showing a dip
	if variance < 0:
	    st.write("Revenues are showing a dip. Budget may need adjustment.")
	else:
	    st.write("Budget holds good even with current revenues.")





	# Function to allocate new budgets based on current revenues
	def allocate_budget_based_on_revenues(initial_budget, current_revenues):
	   return initial_budget+abs(initial_budget-current_revenues)* growth_rate  # Adjust budget based on a percentage (e.g., 10%)

	def allocate_product_budget_based_on_revenues(budget, revenues):
		return budget+abs(budget-revenues)* growth_rate




	# Example: Allocate new budgets based on current revenues
	new_company_budget = allocate_budget_based_on_revenues(initial_company_budget, revenues)
	new_product_budgets = {product: allocate_product_budget_based_on_revenues(budget, revenues) for product, budget in initial_product_budget.items()}

	# Print new budget allocations
	# # st.write(f"New Company Budget Allocation: ${new_company_budget:.2f}")
	# st.write("New Product Budget Allocations:")
	# for product, budget in new_product_budgets.items():
	#     st.write(f"{product}: ${budget:.2f}")

	data12 = {
    'Category': ['Recent| Company Budget','New| Company Budget Allocation'],
    'Allocations': [initial_company_budget, new_company_budget]
	}

	df12 = pd.DataFrame(data12)

	# Create a bar graph using Plotly Express
	fig12 = px.bar(df12, x='Category', y='Allocations', title='Budget Allocation')

	# Streamlit app
	st.title('Budget Allocation')
	st.plotly_chart(fig12, use_container_width=True)    

	################################################










	st.title("If New Coustomers added. Then Revised Revenue Projections")


	# Initial sales budget and revenue projections
	initial_sales_budget = initial_company_budget*0.4  # Example initial sales budget
	initial_revenue_projections = last_budget_forecast  # Example initial revenue projections
	

	

	# Create an input form for new customers
	st.header("Add New Customers")
	num_new_customers = st.number_input("Enter the number of new customers: ", min_value=0, step=1)

	new_customers = []
	for i in range(num_new_customers):
	    st.subheader(f"New Customer {i + 1}")
	    customer_id = st.text_input(f"Enter Customer ID for Customer {i + 1}:")
	    revenue = st.number_input(f"Enter Revenue for Customer {i + 1}:", min_value=0)
	    new_customers.append({'customer_id': customer_id, 'revenue': revenue})

	# Calculate additional revenue from new customers
	additional_revenue = 0
	for customer in new_customers:
	    additional_revenue += customer['revenue']

	if st.button("Updated Revenue Projections"):    

		# Update revenue projections
		revised_revenue_projections = initial_revenue_projections + additional_revenue

		data14 = {
	    'Revenue': ['Recent| Revenue Projections','New| Revised Revenue Projections'],
	    'Projections': [initial_revenue_projections, revised_revenue_projections]
		}

		df14 = pd.DataFrame(data14)

		# Create a bar graph using Plotly Express
		fig14 = px.bar(df14, x='Revenue', y='Projections', title='Updated Revenue Projections')

		# Streamlit app
		st.title('Budget Allocation')
		st.plotly_chart(fig14, use_container_width=True)   

	
	######################
	st.title("If finance Team update Company Budget. Then Revised Projections")
	# Check for budget updates from the finance team
	finance_updated_budget = st.number_input("Enter finance_updated_budget: ", value=200000000, min_value=0)

	if st.button("Updated Revenue Projections",key='k1'): 

		
		initial_company_budget = finance_updated_budget


		st.title("Updated Budget Planning")

		

		# Create a DataFrame from your data
		data1 = pd.DataFrame({'Date': x, 'Forecast': y})

		# Convert the 'Date' column to a datetime data type
		data1['Date'] = pd.to_datetime(data1['Date'])

		# Determine the last date in the data
		last_date = "2021-04-02"

		# Calculate the first date of the last quarter
		last_quarter_start = pd.Timestamp(year=2021, month=1 + 1, day=1)

		# Filter the data for the last quarter
		last_quarter_data = data1[(data1['Date'] >= last_quarter_start) & (data1['Date'] <= last_date)]


		# Calculate the total forecast value for the last quarter
		total_last_quarter_forecast = np.sum(last_quarter_data['Forecast'])


		# Example growth rate 
		# growth_rate = st.number_input("Enter the Growth Rate you want on recent revenue:",value=0.1, min_value=0.0)


		quarterly_sales = df2.resample('Q')['Total Sales'].sum()

		# Get the total sales for the last quarter
		last_quarter_sales = quarterly_sales.iloc[-1]	
		# st.write(last_quarter_sales)        

		revenues = last_quarter_sales  # Example initial revenue
		last_budget_forecast = total_last_quarter_forecast  # Example last budget forecast



		# Create an empty dictionary to store company expenses
		company_expenses = {}

		# Define expense categories and ask users to input expenses
		expense_categories = ['Salaries', 'Utilities', 'Rent', 'Marketing', 'Supplies', 'Other']

		
		company_expenses['Salaries'] = initial_company_budget*0.4
		company_expenses['Utilities'] = initial_company_budget*0.1
		company_expenses['Rent'] = initial_company_budget*0.15
		company_expenses['Marketing'] = initial_company_budget*0.2
		company_expenses['Supplies'] = initial_company_budget*0.1
		company_expenses['Other'] = initial_company_budget*0.05

		data21 = {
	    'Category': expense_categories,
	    'Expense': list(company_expenses.values())
		}

		df21 = pd.DataFrame(data21)

		# Create a bar graph using Plotly Express
		fig21 = px.bar(df21, x='Category', y='Expense', title='Company Expenses')

		# Streamlit app
		st.title('Updated Company Expenses')
		st.plotly_chart(fig21, use_container_width=True)

		#############


		# Create an empty dictionary to store the initial product budget
		initial_product_budget = {}

		# Define the product names
		products = ['Product1', 'Product2', 'Product3']


		initial_product_budget['Product1']=0.15*initial_company_budget
		initial_product_budget['Product2']=0.6*initial_company_budget
		initial_product_budget['Product3']=0.25*initial_company_budget


		# Create an empty dictionary to store the New product budget
		new_product_budget = {}

		# Define the product names
		products = ['Product1', 'Product2', 'Product3']

		# Create input fields for each product's budget
		for product in products:
		    budget = initial_product_budget[str(product)]*growth_rate
		    if budget:
		        new_product_budget[product] = budget

		#initial_product_budget
		data22 = {
	    'Products': list(initial_product_budget.keys()) +list(new_product_budget.keys()),
	    'Allocations': list(initial_product_budget.values()) + list(new_product_budget.values()),
	    'BudgetType': ['Initial'] * len(initial_product_budget) + ['New'] * len(new_product_budget)
		}

		df22 = pd.DataFrame(data22)

		# Create a bar graph using Plotly Express
		fig22 = px.bar(df22, x='Products', y='Allocations', color='BudgetType',title='Product Budget Allocation')

		# Streamlit app
		st.title('Updated Product Budget Allocation')
		st.plotly_chart(fig22, use_container_width=True) 




		company_expenses = sum(company_expenses.values())
		product_expenses = {product: 0 for product in initial_product_budget}

		# Function to check if spending is as per the budget
		def is_spending_within_budget(company_budget, product_budgets, company_expenses, product_expenses):
		    total_company_expenses = sum(product_expenses.values()) + company_expenses
		    return total_company_expenses <= company_budget

		# Function to calculate and track variances from the last budget forecast
		def calculate_variance(revenues, last_forecast):
		    return revenues - last_forecast

		# # Check if spending is within the company and product budgets
		# if is_spending_within_budget(initial_company_budget, initial_product_budget, company_expenses, product_expenses):
		#     st.write("Spending is within the budget.")
		# else:
		#     st.write("Spending exceeds the budget.")

		# Calculate and track variances
		variance = calculate_variance(revenues, last_budget_forecast)




		# Check if the budget still holds good if revenues are showing a dip
		# if variance < 0:
		#     st.write("Revenues are showing a dip. Budget may need adjustment.")
		# else:
		#     st.write("Budget holds good even with current revenues.")



		# Function to allocate new budgets based on current revenues
		def allocate_budget_based_on_revenues(initial_budget, current_revenues):
		   return initial_budget+abs(initial_budget-current_revenues)* growth_rate  # Adjust budget based on a percentage (e.g., 10%)



		# Example: Allocate new budgets based on current revenues
		new_company_budget2 = allocate_budget_based_on_revenues(initial_company_budget, revenues)
		

		data23 = {
	    'Category': ['Expected| Company Budget','Updated| Company Budget Allocation'],
	    'Allocations': [i1, finance_updated_budget]
		}

		df23 = pd.DataFrame(data23)

		# Create a bar graph using Plotly Express
		fig23 = px.bar(df23, x='Category', y='Allocations', title='Budget Allocation')

		# Streamlit app
		st.title('Updated Budget Allocation')
		st.plotly_chart(fig23, use_container_width=True)    

		################################################

	 







def feature_1():

	df_new = df.groupby('Invoice Date').agg({
		'Price per Unit': 'sum',
    	'Units Sold': 'sum',
    	'Total Sales': 'sum',
    	'Operating Profit': 'sum',
    	'Operating Margin': 'mean'}).reset_index()


	df_new['Invoice Date'] = pd.to_datetime(df_new['Invoice Date'])

	

	selected_column = st.sidebar.selectbox("Select a Feature:", ['Total Sales','Price per Unit','Units Sold','Operating Profit','Operating Margin'])

	st.title("Time Period Selector")

	time_period = st.sidebar.selectbox("Select a Time Period:", ["Monthly", "Quarterly", "Yearly","Daily"])


	st.title("Date Range should be in between Jan,2020-Dec,2021")

	# Sidebar for user input
	start_date = st.sidebar.date_input("Select a Start Date")
	end_date = st.sidebar.date_input("Select an End Date")

	# Check if the start date is before the end date
	if start_date <= end_date:
	  st.write(f"Start Date: {start_date}")
	  st.write(f"End Date: {end_date}")
	  
	  # Convert dates to datetime objects for filtering
	  start_date = datetime.combine(start_date, datetime.min.time())
	  end_date = datetime.combine(end_date, datetime.max.time())
	  

	



	if time_period == "Monthly":
	  dates = pd.date_range(start=start_date,end=end_date,freq="MS")
	  # monthly_df = df[df['date'].isin(monthly_dates)].copy()
	  # dates_d=monthly_dates
	elif time_period == "Quarterly":
	  dates = pd.date_range(start=start_date,end=end_date,freq="QS")
	  # quarterly_df = df[df['date'].isin(quarterly_dates)].copy()
	  # dates_d=quarterly_dates
	elif time_period == "Yearly":
	  dates = pd.date_range(start=start_date,end=end_date,freq="Y")
	elif time_period == "Daily":
	  dates = pd.date_range(start=start_date,end=end_date,freq="D")



	st.title("Click below button to Plot")

	if st.button("Plot Graph"):

		filtered_df=df_new[df_new['Invoice Date'].isin(dates)]


		def plot(x1,y1):

				
			trace1 = go.Scatter(x=x1,y=y1, mode='lines+markers', name='Actual')
			layout = go.Layout(title="Actual")
			fig = go.Figure(data=[trace1], layout=layout)
			st.plotly_chart(fig)


		plot(filtered_df['Invoice Date'],filtered_df[selected_column])











def feature_2():

	df_new = df.groupby('Invoice Date').agg({
		'Price per Unit': 'sum',
    	'Units Sold': 'sum',
    	'Total Sales': 'sum',
    	'Operating Profit': 'sum',
    	'Operating Margin': 'mean'}).reset_index()


	df_new['Invoice Date'] = pd.to_datetime(df_new['Invoice Date'])

	selected_column = st.sidebar.selectbox("Select a Feature:", ['Total Sales','Price per Unit','Units Sold','Operating Profit','Operating Margin'])

	# Display the selected column from the DataFrame
	# st.write(f"Selected Column: {selected_column}")
	# st.write(df[selected_column])

	############ 1 ###############

	st.title("Date Range should be in between Jan,2020-Dec,2021")

	# Sidebar for user input
	start_date = st.sidebar.date_input("First Duration | Select a Start Date")
	end_date = st.sidebar.date_input("First Duration | Select an End Date")

	# Check if the start date is before the end date
	if start_date <= end_date:
	  st.write(f"Start Date: {start_date}")
	  st.write(f"End Date: {end_date}")
	  
	  # Convert dates to datetime objects for filtering
	  start_date = datetime.combine(start_date, datetime.min.time())
	  end_date = datetime.combine(end_date, datetime.max.time())
	  

	st.title("First Duration | Time Period Selector")

	time_period = st.sidebar.selectbox("First Duration | Select a Time Period:", ["Monthly", "Quarterly", "Yearly","Daily"],key="s0")

	if time_period == "Monthly":
	  	dates = pd.date_range(start=start_date,end=end_date,freq="MS")
	elif time_period == "Quarterly":
	  	dates = pd.date_range(start=start_date,end=end_date,freq="QS")
	elif time_period == "Yearly":
	  	dates = pd.date_range(start=start_date,end=end_date,freq="Y")
	elif time_period == "Daily":
		dates = pd.date_range(start=start_date,end=end_date,freq="D")  
	  

	############# 2 ##################  

	st.title("Date Range should be in between Jan,2020-Dec,2021")

	# Sidebar for user input
	start_date1 = st.sidebar.date_input("Second Duration | Select a Start Date",key="sd1")
	end_date1 = st.sidebar.date_input("Second Duration | Select an End Date",key='sd2')

	# Check if the start date is before the end date
	if start_date1 <= end_date1:
	  st.write(f"Start Date: {start_date1}")
	  st.write(f"End Date: {end_date1}")
	  
	  # Convert dates to datetime objects for filtering
	  start_date1 = datetime.combine(start_date1, datetime.min.time())
	  end_date1 = datetime.combine(end_date1, datetime.max.time())
	  

	st.title("Time Period Selector")

	time_period1 = st.sidebar.selectbox("Second Duration | Select a Time Period:", ["Monthly", "Quarterly", "Yearly","Daily"],key="s1")

	if time_period == "Monthly":
	  	dates1 = pd.date_range(start=start_date1,end=end_date1,freq="MS")
	elif time_period == "Quarterly":
	  	dates1 = pd.date_range(start=start_date1,end=end_date,freq="QS")
	elif time_period == "Yearly":
	  	dates1 = pd.date_range(start=start_date1,end=end_date1,freq="Y")
	elif time_period == "Daily":
		dates1 = pd.date_range(start=start_date1,end=end_date1,freq="D")  
        



    

	st.title("Click below button to Compre Plots")

	# Create a button
	if st.button("Plot Graphs"):

		filtered_df1=df_new[df_new['Invoice Date'].isin(dates)]
		filtered_df2=df_new[df_new['Invoice Date'].isin(dates1)]
	  # Function to plot a simple graph
		def plot(x1,y1):
		   trace1 = go.Scatter(x=x1, y=y1, mode='lines+markers', name='Actual')
		   layout = go.Layout(title="Actual")
		   fig = go.Figure(data=[trace1], layout=layout)
		   # pyo.iplot(fig)
		   st.plotly_chart(fig,use_container_width=True)


		# Call the function
		col1, col2= st.columns(2)

		with col1:
			plot(filtered_df1['Invoice Date'],filtered_df1[selected_column])  
		
		with col2:
			plot(filtered_df2['Invoice Date'],filtered_df2[selected_column])       







def feature_forecast():

	st.title("Total Sales Forecasting")

	df['Date'] = pd.to_datetime(df['Invoice Date'])

	df2= df[['Date', 'Total Sales']]
	df2 = df2.groupby('Date')[['Total Sales']].sum().reset_index()

	df2.set_index('Date', inplace=True)

	model=sm.tsa.statespace.SARIMAX(df2['Total Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
	results=model.fit()
	
	start_date = st.sidebar.date_input("Select a Start Date")
	end_date = st.sidebar.date_input("Select an End Date")

  
	if start_date <= end_date:
	   st.write(f"Start Date: {start_date}")
	   st.write(f"End Date: {end_date}")
	     
	   start_date = pd.to_datetime(start_date)
	   end_date = pd.to_datetime(end_date)

	frequency = st.sidebar.selectbox("Select a Time Period Frequency:", ["Monthly", "Quarterly", "Yearly","Daily"])


	if frequency=="Monthly":
	    frequency='MS'
	elif frequency=="Quarterly":
	    frequency='QS'
	elif frequency=="Yearly":
	    frequency='Y'
	elif frequency=="Daily":
	    frequency='D'    
	    
	if st.button("Plot Forecaste"):

		index_future_dates=pd.date_range(start=start_date,end=end_date,freq=frequency)
		#print(index_future_dates
		C=len(index_future_dates)
		pred=results.predict(start=len(df2),end=len(df2)+C-1,typ='levels').rename('ARIMA Predictions')
		#print(comp_pred)
		pred.index=index_future_dates
		# print(pred)

		fig = go.Figure()

		fig.add_trace(go.Scatter(x=pred.index, y=pred, mode='lines', name='Forecast Sales'))

		# Update the layout of the plot
		fig.update_layout(title='Forecasted Sales ',
		                  xaxis_title='Date',
		                  yaxis_title='Total Sales')

		# Show the plot
		st.plotly_chart(fig)





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
        options=["Home","Predictions","Plot a Feature","Compare Two Durations of a Feature","Forecast a Feature", "Budget Planning","Recommendations","Assistant"],
        icons=["house"],
        menu_icon="cast",
        default_index=0
    )
 if selected=="Home":
    
    Home()
    
 if selected=="Plot a Feature":
    st.subheader(f"Page: {selected}")
    feature_1()
    

 if selected=="Compare Two Durations of a Feature":
    st.subheader(f"Page: {selected}")
    feature_2()      

 if selected=="Forecast a Feature":
    st.subheader(f"Page: {selected}")
    feature_forecast()

 if selected=="Predictions":
    st.subheader(f"Page: {selected}")
    feature_predictions()

 if selected=="Budget Planning":
    st.subheader(f"Page: {selected}")
    qqt()
 if selected=="Recommendations":
    st.subheader(f"Page: {selected}")
    recommendations()  
 if selected=="Assistant":
    st.subheader(f"Page: {selected}")
    assistant()  

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

    
