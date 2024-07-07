

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from streamlit_option_menu import option_menu
import pickle


st.set_page_config(page_title="SINGAPORE RESALE FLAT PRICES PREDICTING", page_icon=":anchor:", layout="wide", menu_items=None)

select= option_menu(menu_title=None,
                    options = ["Home","Price Prediction"],
                    default_index=0,
                    orientation="horizontal",
                    styles={
            "container": {"padding": "0!important", "background-color": "white","size":"cover"},
            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "green"},
            "nav-link-selected": {"background-color": "green"}
        } )
s_year= ["2015", "2016", "2017", "2018", "2019", "2020", "2021","2022", "2023", "2024"]

s_town= ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
'TOA PAYOH', 'WOODLANDS', 'YISHUN']

s_flat_type=['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM','MULTI-GENERATION']

s_flat_model=['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen']

def town_mapping(town_map):
    town_dict = {'ANG MO KIO': 0,'BEDOK': 1,'BISHAN': 2,'BUKIT BATOK': 3,'BUKIT MERAH': 4,'BUKIT PANJANG': 5,'BUKIT TIMAH': 6,
        'CENTRAL AREA': 7,'CHOA CHU KANG': 8,'CLEMENTI': 9,'GEYLANG': 10,'HOUGANG': 11,'JURONG EAST': 12,'JURONG WEST': 13,
        'KALLANG/WHAMPOA': 14,'MARINE PARADE': 15,'PASIR RIS': 16,'PUNGGOL': 17,'QUEENSTOWN': 18,'SEMBAWANG': 19,'SENGKANG': 20,'SERANGOON': 21,
        'TAMPINES': 22,'TOA PAYOH': 23,'WOODLANDS': 24,'YISHUN': 25
    }
    return town_dict.get(town_map, -1)  # Return -1 if town_map not found


def flat_type_mapping(flt_type):
    flat_type_dict = {'3 ROOM': 2,'4 ROOM': 3,'5 ROOM': 4,'2 ROOM': 1,'EXECUTIVE': 5,'1 ROOM': 0,'MULTI-GENERATION': 6
    }
    return flat_type_dict.get(flt_type, -1)  # Return -1 if flt_type not found


def flat_model_mapping(fl_m):
    flat_model_dict = {'Improved': 5,'New Generation': 12,'Model A': 8,'Standard': 17,'Simplified': 16,'Premium Apartment': 13,
                       'Maisonette': 7,'Apartment': 3,'Model A2': 10,'Type S1': 19,'Type S2': 20,'Adjoined flat': 2,'Terrace': 18,'DBSS': 4,
                       'Model A-Maisonette': 9,'Premium Maisonette': 15,'Multi Generation': 11,'Premium Apartment Loft': 14,
                        'Improved-Maisonette': 6,'2-room': 0,'3Gen': 1
    }
    return flat_model_dict.get(fl_m, -1)  # Return -1 if fl_m not found


def predict_price(year,town,flat_type,flr_area_sqm,flat_model,stry_start,stry_end,re_les_year,
              re_les_month):
    
    year_1= int(year)
    town_2= town_mapping(town)
    flt_ty_2= flat_type_mapping(flat_type)
    flr_ar_sqm_1= int(flr_area_sqm)
    flt_model_2= flat_model_mapping(flat_model)
    str_str= np.log(int(stry_start))
    str_end= np.log(int(stry_end))
    rem_les_year= int(re_les_year)
    rem_les_month= int(re_les_month)


    with open("C:/Users/Happy/Desktop/capstone/singapore resale/price_prediction1.pkl","rb") as f:
        regg_model= pickle.load(f)

    user_data = np.array([[year_1,town_2,flt_ty_2,flr_ar_sqm_1,
                           flt_model_2,str_str,str_end,rem_les_year,rem_les_month]])
    y_pred_1 = regg_model.predict(user_data)
    price= np.exp(y_pred_1[0])

    return round(price)


if select == "Home":

    st.title(":green[SINGAPORE RESALE FLAT PRICES PREDICTING]")
    st.write("")

    st.write(":green[**The goal of this project:**]")
    st.write(" ")
    st.write("To predict the resale prices of HDB Flats according to their specifications such as the floor area, storey, location etc.")
    st.write("Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")
    st.write("Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.")
    st.write("Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.")
    st.write("Exploring and optimizing hyperparameters, you can significantly improve the predictive power and robustness of your machine learning models.")
    st.write("Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.")
    st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")
    

if select == "Price Prediction":

    st.title(":green[SINGAPORE RESALE FLAT PRICES PREDICTING]")
    st.write("")

    col1,col2= st.columns(2)
    with col1:

        year= st.selectbox("Select the Year",s_year)
        
        town= st.selectbox("Select the Town", s_town)
        
        flat_type= st.selectbox("Select the Flat Type", s_flat_type )
        
        flr_area_sqm= st.text_input("Enter the Value of Floor Area sqm (Min: 31 / Max: 280")

        flat_model= st.selectbox("Select the Flat Model", s_flat_model)
        
    with col2:

        stry_start= st.text_input("Enter the Upper Storey ")

        stry_end= st.text_input("Enter the Lower Storey ")

        re_les_year= st.text_input("Enter the Value of Remaining Lease Year (Min: 42 / Max: 97)")

        re_les_month= st.text_input("Enter the Value of Remaining Lease Month (Min: 0 / Max: 11)")
    
    with col1:

        button= st.button("Predict the Price", use_container_width= True)

    if button:

        
        pre_price= predict_price(year, town, flat_type, flr_area_sqm, flat_model,
                        stry_start, stry_end, re_les_year, re_les_month)
        
        formatted_price = '{:,}'.format(pre_price)
        
        st.title(f':green[**Predicted Selling Price :**] :green[S$] **:green[{formatted_price}]**')


