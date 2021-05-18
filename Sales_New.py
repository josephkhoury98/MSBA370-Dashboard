import pandas as pd
import numpy as np  
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
import matplotlib.image as mpimg

#------------------------------ Data collection ---------------------------------
sales = pd.read_csv('https://github.com/josephkhoury98/MSBA370-Dashboard/blob/main/SALES_SALES.csv',error_bad_lines=False)
sales = sales.rename({'Sales Category': 'Sales_Category'}, axis=1)
sales.info()
rent = pd.read_csv('https://github.com/josephkhoury98/MSBA370-Dashboard/blob/main/RENT_RENT.csv', error_bad_lines=False)
rent = rent.rename({'Sales Category': 'Sales_Category'}, axis=1)
rent.info()

missing_values_count = pd.DataFrame({'Null':sales.isnull().sum()})
total = len(sales)
percentage_null = round((missing_values_count['Null']/total)*100,2)
missing_values_count['Percentage'] = percentage_null
missing_values_count.sort_values(by='Null', ascending = False)

missing_rent_values_count = pd.DataFrame({'Null':rent.isnull().sum()})
total = len(rent)
percentage_null = round((missing_rent_values_count['Null']/total)*100,2)
missing_rent_values_count['Percentage'] = percentage_null
missing_rent_values_count.sort_values(by='Null', ascending = False)

#All null values are in features that we don't need, thus we will ignore them

#-----------------------------------------Format of the page------------------------------------------------------
st.set_page_config(layout='wide')
#st.markdown(f"<h1 style='text-align:Center; font-family:arial;' >{'<b>MSBA 370 - DDDM</b>'}</h1>", unsafe_allow_html=True)

#LOGO_IMAGE = mpimg.imread('C:/Users/Admin/Desktop/Project/Majid Al Futtaim logo.png')
st.markdown(
    """
    <style>
      .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
        padding-top: 75px !important;
        text-align:Center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="container">
        <p class="logo-text">MSBA 370</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 =st.beta_columns([47.5,5,47.5])
#------------------------------------------Data per Category -----------------------------------------------------
rent_category = rent.groupby(rent['Sales_Category']).sum()
rent_category = pd.DataFrame(rent_category.to_records())
rent_category = rent_category.drop(['Sales_Category', 'GLA'], axis=1)
rent_category = pd.DataFrame(rent_category.to_records())
sales_category = sales.groupby(sales['Sales_Category']).sum()
sales_category = pd.DataFrame(sales_category.to_records())
sales_category = pd.concat([sales_category, rent_category.reindex(sales_category.index)], axis=1)
sales_category['Sales by SQM'] = sales_category['Sales']/sales_category['GLA']
sales_category['Rent by SQM'] = sales_category['Rent']/sales_category['GLA']
sales_category = sales_category.sort_values(by='Sales by SQM', ascending=False)
sales_category.info()

missing_cat_values_count = pd.DataFrame({'Null':sales_category.isnull().sum()})
total = len(sales_category)
percentage_null = round((missing_cat_values_count['Null']/total)*100,2)
missing_cat_values_count['Percentage'] = percentage_null
missing_cat_values_count.sort_values(by='Null', ascending = False)

#No missing values thus no need to impute or do anything

rent_sales_persqm = make_subplots(specs=[[{"secondary_y": True}]])
rent_sales_persqm.add_trace(go.Scatter(x=sales_category['Sales_Category'], y=sales_category['Rent by SQM'], name = 'Rent per SQM'),secondary_y=True,)
rent_sales_persqm.add_trace(go.Bar(x=sales_category['Sales_Category'], y=sales_category['Sales by SQM'], name= "Categories"),secondary_y=False,)
#rent_sales_persqm.update_layout(title_text="Double Y Axis Example")
rent_sales_persqm.update_layout(xaxis={'visible': False, 'showticklabels': False})
rent_sales_persqm.update_layout(template="simple_white")
#rent_sales_persqm.update_yaxes(title_text="yaxis title")
#rent_sales_persqm.update_yaxes(title_text="Sales (in USD)", secondary_y=False)
rent_sales_persqm.update_yaxes(title_text="<b>Rent </b>per SQM", secondary_y=True)
rent_sales_persqm.update_yaxes(range=[0,500], secondary_y=True)
rent_sales_persqm.update_yaxes(title_text="<b>Sales </b>per SQM", secondary_y=False)
#----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------

sales_area = px.scatter(sales_category,x= sales_category['GLA'],y= sales_category['Sales'], trendline='ols', trendline_color_override = 'lightgrey' ,
 template="simple_white", labels={'GLA' : 'Area (in SQM)'}, hover_name=sales_category['Sales_Category'] )
#----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------

#gla, ax5 = plt.subplots()
#sns.regplot(data = sales_category, x= sales_category['GLA'],y= sales_category['Sales'], ci=None, line_kws={"color": "lightgrey"}, scatter_kws={"color": "darkblue"})
#plt.text(48300, 0.02*10**8, "O Entertainment", horizontalalignment='left', size='medium', color='darkred', weight='semibold')
#plt.text(153000, 0.08*10**8, "O Cinema", horizontalalignment='left', size='medium', color='darkred', weight='semibold')
#trying to cheat the system with an O to highlight a point
#plt.xlim(0, 160000)
#st.pyplot(gla)



#-------------------------------------------Yearly and Monthly Sales ----------------------------------------------
#Calculate Monthly sales & putting result in a new dataframe
sales_monthly = pd.pivot_table(sales, values='Sales',index = 'Date', aggfunc=np.sum)
sales_monthly = pd.DataFrame(sales_monthly.to_records())
sales_monthly.info()
sales_monthly['Date']= pd.to_datetime(sales_monthly['Date'])
sales_monthly = sales_monthly.sort_values(by='Date')



monthly_sales_graph = make_subplots(specs=[[{"secondary_y": True}]])
monthly_sales_graph.add_trace(go.Scatter(x=sales_monthly['Date'].iloc[0:12], y=sales_monthly['Sales'].iloc[0:12], name="2017"),secondary_y=False,)
monthly_sales_graph.add_trace(go.Scatter(x=sales_monthly['Date'].iloc[0:12], y=sales_monthly['Sales'].iloc[12:24], name="2018"),secondary_y=False,)
#monthly_sales_graph.update_layout(title_text="Double Y Axis Example")
monthly_sales_graph.update_layout(xaxis={'visible': False, 'showticklabels': False})
monthly_sales_graph.update_layout(template="simple_white")
#monthly_sales_graph.update_yaxes(title_text="yaxis title")
monthly_sales_graph.update_yaxes(title_text="Sales (in USD)", secondary_y=False)

#----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------


#Calculte yearly sales & result in a new dataframe
sales_yearly = sales_monthly.groupby(sales_monthly['Date'].dt.year).sum()
sales_yearly = pd.DataFrame(sales_yearly.to_records())
sales_yearly['Date'] = pd.to_datetime(sales_yearly['Date'],format='%Y')
sales_yearly.info()

yearly_bar= px.bar(sales_yearly, sales_yearly['Date'].dt.year, sales_yearly['Sales'], color = sales_yearly['Date'])
yearly_bar.update_yaxes(title_text="Sales in (USD)", secondary_y=False)
yearly_bar.update_xaxes(title_text="Years")
yearly_bar.update_layout(template="simple_white")

#----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------



#st.markdown(f"<h3 style='text-align:left; font-family:arial;' >{'<b>Previous Telemarketing Campaign Metrics</b>'}</h3>", unsafe_allow_html=True)
col1.markdown("""<div style = "background-color: rgb(73,145,219)"><p style = "color: rgb(255,255,255); font-size:20px"><b>Mall Total Sales</b></p></div>""", unsafe_allow_html = True)
col1.markdown(f"<h4 style='text-align:center; font-family:arial;' >{'Yearly Sales across the mall'}</h4>", unsafe_allow_html=True)
col1.plotly_chart(yearly_bar)
col1.markdown("""<div style = "background-color: rgb(235,126,37)"><p style = "color: rgb(255,255,255); font-size:20px"><b>Areas of Interest</b></p></div>""", unsafe_allow_html = True)
col1.markdown(f"<h4 style='text-align:center; font-family:arial;' >{'Sales and Rent per SQM'}</h4>", unsafe_allow_html=True)
col1.plotly_chart(rent_sales_persqm)

col3.markdown(f"<h3 style='text-align:left; font-family:arial;' >{'<b>  </b>'}</h3>", unsafe_allow_html=True)
col3.markdown(f"<h3 style='text-align:left; font-family:arial;' >{'<b>  </b>'}</h3>", unsafe_allow_html=True)
col3.markdown(f"<h4 style='text-align:center; font-family:arial;' >{'Monthly Sales across the mall'}</h4>", unsafe_allow_html=True)
col3.plotly_chart(monthly_sales_graph)
col3.markdown(f"<h3 style='text-align:left; font-family:arial;' >{'<b>  </b>'}</h3>", unsafe_allow_html=True)
col3.markdown(f"<h3 style='text-align:left; font-family:arial;' >{'<b>  </b>'}</h3>", unsafe_allow_html=True)
col3.markdown(f"<h4 style='text-align:center; font-family:arial;' >{'Sales per Category vs Area'}</h4>", unsafe_allow_html=True)
col3.plotly_chart (sales_area)



#--------------------------------------Prediction of Sales------------------------
pred_data = sales_category
pred_data = pred_data.drop('index', axis=1)
categorical_df = pred_data[['Sales_Category']]
numerical_df = pred_data[['GLA', 'Sales', 'Rent', 'Rent by SQM']]
dummy = pd.get_dummies(categorical_df)
encoded_df= pd.concat ([numerical_df, dummy], axis=1)
encoded_df.info()

x_base = encoded_df.drop('Sales', axis=1)
y_base = encoded_df['Sales'].copy()

X_train, X_test, y_train, y_test = train_test_split( x_base, y_base, test_size=0.2, random_state=42)

linReg = LinearRegression()
linReg.fit(X_train, y_train)
score = []
cross_val_scores = cross_val_score(linReg, X_train, y_train, scoring= "neg_mean_squared_error", cv=5)
cross_val_rmse = np.sqrt(-cross_val_scores)

def display_scores(scores):
    print("Scores:", np.round(scores))
    print("Mean: ", np.round(scores.mean()))
    print("Stdev:", np.round(scores.std()))

display_scores(cross_val_rmse) 

col1.markdown("""<div style = "background-color: rgb(73,145,219)"><p style = "color: rgb(255,255,255); font-size:20px"><b>Mall Sales Prediction</b></p></div>""", unsafe_allow_html = True)
col3.markdown(f"<h3 style='text-align:left; font-family:arial;' >{'<b> </b>'}</h3>", unsafe_allow_html=True)

cat=col1.selectbox('Select a Category:',('Arabian Perfumes','Cafes','Childrens Fashion','Cinema', 'Department Stores','Electronics & Gadgets & Accessories',
'Entertainment','Entertainment - Other', 'Fashion Accessories', 'Female Fashion', 'Food Court','Footwear','General Store', 'Health & Nutrition','Home Furnishings','Hypermarket',
'Jewellery','Lingerie & Swimwear', 'Luggage', 'Mens Fashion', 'Mobile Phone & Accessories','Music', 'Optometrist & Sun Glasses','Perfumes & Cosmetics', 'Pharmacy', 'Restaurants',
'Services', 'Speciality Food','Sports & Leisure','Toys & Hobbies','Unisex Fashion','Watches'))

Sales_Category_ArabianPerfumes = 0
Sales_Category_Cafes = 0
Sales_Category_ChildrensFashion = 0
Sales_Category_Cinema = 0
Sales_Category_DepartmentStores = 0
Sales_Category_Electronics_Gadgets_Accessories = 0
Sales_Category_Entertainment = 0
Sales_Category_Entertainment_Other = 0
Sales_Category_FashionAccessories = 0
Sales_Category_FemaleFashion = 0
Sales_Category_FoodCourt = 0
Sales_Category_Footwear = 0
Sales_Category_GeneralStore = 0
Sales_Category_Health_Nutrition = 0
Sales_Category_HomeFurnishings = 0
Sales_Category_Hypermarket = 0
Sales_Category_Jewellery = 0
Sales_Category_Lingerie_Swimwear = 0
Sales_Category_Luggage = 0
Sales_Category_MensFashion = 0
Sales_Category_MobilePhone_Accessories = 0
Sales_Category_Music = 0
Sales_Category_Optometrist_SunGlasses = 0
Sales_Category_Perfumes_Cosmetics = 0
Sales_Category_Pharmacy = 0
Sales_Category_Restaurants = 0
Sales_Category_Services = 0
Sales_Category_SpecialityFood = 0
Sales_Category_Sports_Leisure = 0
Sales_Category_Toys_Hobbies = 0
Sales_Category_UnisexFashion = 0
Sales_Category_Watches = 0
if cat == 'Arabian Perfumes':
    Sales_Category_ArabianPerfumes = 1
elif cat == 'Cafes':
    Sales_Category_Cafes = 1
elif cat == 'Childrens Fashion': 
    Sales_Category_ChildrensFashion = 1
elif cat == 'Cinema':
    Sales_Category_Cinema = 1
elif cat == 'Department Stores':
    Sales_Category_DepartmentStores = 1
elif cat == 'Electronics & Gadgets & Accessories':
    Sales_Category_Electronics_Gadgets_Accessories = 1
elif cat == 'Entertainment':
    Sales_Category_Entertainment = 1
elif cat == 'Entertainment - Other':
    Sales_Category_Entertainment_Other = 1
elif cat == 'Fashion Accessories':
    Sales_Category_FashionAccessories = 1
elif cat == 'Female Fashion':
    Sales_Category_FemaleFashion = 1
elif cat == 'Food Court':
    Sales_Category_FoodCourt = 1
elif cat == 'Footwear':
    Sales_Category_Footwear = 1
elif cat == 'General Store':
    Sales_Category_GeneralStore = 1
elif cat == 'Health & Nutrition':
    Sales_Category_Health_Nutrition = 1
elif cat == 'Home Furnishings':
    Sales_Category_HomeFurnishings = 1
elif cat == 'Hypermarket':
    Sales_Category_Hypermarket = 1
elif cat == 'Jewellery':
    Sales_Category_Jewellery = 1
elif cat == 'Lingerie & Swimwear':
    Sales_Category_Lingerie_Swimwear = 1
elif cat == 'Luggage':
    Sales_Category_Luggage = 1
elif cat == 'Mens Fashion':
    Sales_Category_MensFashion = 1
elif cat == 'Mobile Phone & Accessories':
    Sales_Category_MobilePhone_Accessories = 1
elif cat == 'Music':
    Sales_Category_Music = 1
elif cat == 'Optometrist & Sun Glassess':
    Sales_Category_Optometrist_SunGlasses = 1
elif cat == 'Perfumes & Cosmetics':
    Sales_Category_Perfumes_Cosmetics = 1
elif cat == 'Pharmacy':
    Sales_Category_Pharmacy = 1
elif cat == 'Restaurants':
    Sales_Category_Restaurants = 1
elif cat == 'Services':
    Sales_Category_Services = 1
elif cat == 'Speciality Food':
    Sales_Category_SpecialityFood = 1
elif cat == 'Sports & Leisure':
    Sales_Category_Sports_Leisure = 1
elif cat == 'Toys & Hobbies':
    Sales_Category_Toys_Hobbies = 1
elif cat == 'Unisex Fashion':
    Sales_Category_UnisexFashion = 1
elif cat == 'Watches':
    Sales_Category_Watches = 1



GLA_input = col3.text_input("Category Surface Area", "Type here")
Rent_input = col1.text_input("Rent per Category", "Type here")
RentBySQM = col3.text_input("Rent per SQM", "Type here")
def predict_sales(GLA, Rent, RentBySQM,Sales_Category_ArabianPerfumes,Sales_Category_Cafes,Sales_Category_ChildrensFashion,Sales_Category_Cinema,Sales_Category_DepartmentStores,Sales_Category_Electronics_Gadgets_Accessories,Sales_Category_Entertainment,Sales_Category_Entertainment_Other,Sales_Category_FashionAccessories,Sales_Category_FemaleFashion,Sales_Category_FoodCourt,Sales_Category_Footwear,Sales_Category_GeneralStore,Sales_Category_Health_Nutrition,Sales_Category_HomeFurnishings,Sales_Category_Hypermarket,Sales_Category_Jewellery,Sales_Category_Lingerie_Swimwear,Sales_Category_Luggage,Sales_Category_MensFashion,Sales_Category_MobilePhone_Accessories,Sales_Category_Music,Sales_Category_Optometrist_SunGlasses,Sales_Category_Perfumes_Cosmetics,Sales_Category_Pharmacy,Sales_Category_Restaurants,Sales_Category_Services,Sales_Category_SpecialityFood,Sales_Category_Sports_Leisure,Sales_Category_Toys_Hobbies,Sales_Category_UnisexFashion,Sales_Category_Watches):
    prediction = linReg.predict([[GLA, Rent, RentBySQM,Sales_Category_ArabianPerfumes,Sales_Category_Cafes,Sales_Category_ChildrensFashion,Sales_Category_Cinema,Sales_Category_DepartmentStores,Sales_Category_Electronics_Gadgets_Accessories,Sales_Category_Entertainment,Sales_Category_Entertainment_Other,Sales_Category_FashionAccessories,Sales_Category_FemaleFashion,Sales_Category_FoodCourt,Sales_Category_Footwear,Sales_Category_GeneralStore,Sales_Category_Health_Nutrition,Sales_Category_HomeFurnishings,Sales_Category_Hypermarket,Sales_Category_Jewellery,Sales_Category_Lingerie_Swimwear,Sales_Category_Luggage,Sales_Category_MensFashion,Sales_Category_MobilePhone_Accessories,Sales_Category_Music,Sales_Category_Optometrist_SunGlasses,Sales_Category_Perfumes_Cosmetics,Sales_Category_Pharmacy,Sales_Category_Restaurants,Sales_Category_Services,Sales_Category_SpecialityFood,Sales_Category_Sports_Leisure,Sales_Category_Toys_Hobbies,Sales_Category_UnisexFashion,Sales_Category_Watches]])
    return prediction

results=""
if col1.button('Predict'):
    results = predict_sales(GLA_input, Rent_input, RentBySQM,Sales_Category_ArabianPerfumes,Sales_Category_Cafes,Sales_Category_ChildrensFashion,Sales_Category_Cinema,Sales_Category_DepartmentStores,Sales_Category_Electronics_Gadgets_Accessories,Sales_Category_Entertainment,Sales_Category_Entertainment_Other,Sales_Category_FashionAccessories,Sales_Category_FemaleFashion,Sales_Category_FoodCourt,Sales_Category_Footwear,Sales_Category_GeneralStore,Sales_Category_Health_Nutrition,Sales_Category_HomeFurnishings,Sales_Category_Hypermarket,Sales_Category_Jewellery,Sales_Category_Lingerie_Swimwear,Sales_Category_Luggage,Sales_Category_MensFashion,Sales_Category_MobilePhone_Accessories,Sales_Category_Music,Sales_Category_Optometrist_SunGlasses,Sales_Category_Perfumes_Cosmetics,Sales_Category_Pharmacy,Sales_Category_Restaurants,Sales_Category_Services,Sales_Category_SpecialityFood,Sales_Category_Sports_Leisure,Sales_Category_Toys_Hobbies,Sales_Category_UnisexFashion,Sales_Category_Watches )
    col1.success('In the next 2 years, {} category will have ${} of sales '.format(cat, results))
