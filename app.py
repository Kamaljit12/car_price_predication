import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64



## ========================= models==================
# rf_model = pickle.load(open("RandomForest.pkl", 'rb'))
knn_model = pickle.load(open("K-Neighbors.pkl", 'rb'))
scaler_model = pickle.load(open("scaler.pkl", "rb"))
ohe_encoder_model = pickle.load(open("ohe_encoder.pkl", "rb"))



# =======app header=================================================

c1, c2, c3 = st.columns(3)
with c1:
    c1.image('img/carLogo.png', width=150)

with c3:
    c3.markdown("<h2 style=color:#C10789;>Car Price Prediction</h2>", unsafe_allow_html=True)

## load dataframe
df = pd.read_csv("trained_car_dataset.csv")



side_options = ['Price Prediction', 'About Datasets' ,"Analysis", 'About Me']
  
side_bar_selected = st.sidebar.selectbox("Menu", options=side_options)

if  side_bar_selected == 'Price Prediction':
    ## ================input section =====================================
    st.header("Predict Car Price!")

    col1, col2, col3 = st.columns(3)

    with col1:

        model_options = df['Model'].unique()
        ## user input
        model_selected = col1.selectbox("Select Car Model", options=model_options)

    with col2:
        interior_options = df['Leather_interior'].unique()
        interior_selected = col2.selectbox("Leather Interior", options=interior_options)

    with col3:
        fuel_options = df['Fuel_type'].unique()
        fuel_selected = col3.selectbox("Fuel Type", fuel_options)
        

    col1, col2 = st.columns(2)

    with col1:
        mileagest_selected = st.number_input("Covered Distances by Car")

    with col2:
        gear_box_options = df['Gear_box_type'].unique()
        gear_box_selected = st.selectbox("Gear Box Type", gear_box_options)

    col1, col2 = st.columns(2)
    with col1:
        airbag_selected = st.select_slider("Air Bags", range(1, 17))

    with col2:
        car_age_selected = st.select_slider("How Old is Your Car ?", range(1, 19))


    ##==================== data preperation =========================


    cat_col = [model_options, interior_selected, fuel_selected, gear_box_selected]
    num_col = [mileagest_selected, airbag_selected, car_age_selected]

    encoded_data = ohe_encoder_model.transform([[model_selected, interior_selected, fuel_selected, gear_box_selected]]).toarray()
    scaled_data = scaler_model.transform([[mileagest_selected, airbag_selected, car_age_selected]])


    main_df = pd.concat((pd.DataFrame(encoded_data), pd.DataFrame(scaled_data)), axis=1)
    prediction = knn_model.predict(main_df)

    if st.button("Predict"):
        st.success(np.round(prediction[0], 2))

# ===========================================================================================

elif side_bar_selected == 'About Datasets':

    ## ----------sample data
    if st.button("Show sample data"):
        st.dataframe(df.sample(5))

    ## ----------------EDA-----------------

    st.subheader("Some important satas of the datasets")
    st.dataframe(np.round(df.describe()))

    ## numerical features
    categorical_features = [col for col in df.columns if df[col].dtype=='O']
    numerical_features = [col for col in df.columns if df[col].dtype!='O']



    st.subheader("Count of all categorical features")
    for col in categorical_features:
        size =  len(df[col].unique())
        st.text(f"Total --> {col} - {size}")

    ##-----som important information about numerical features
    st.subheader("Min and Max values of the categorical features")
    for col in numerical_features:
        min = df[col].min()
        max = df[col].max()
        st.text(f"Minimum of {col} is = {min} and maximun is = {max}")

    st.subheader("Total numbers of features")
    for col in df.columns:
        st.text(col)

    # =================== visualization ================


    # st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader("Density plot of each numerical features")
    for col in numerical_features:
        sns.displot(data=df, x=col, kind='kde', hue='Leather_interior')

        st.pyplot()


    st.subheader("Top 5 most frequent from each features bar chart")
    for col in categorical_features:
        sns.barplot(df[col].value_counts(ascending=False).head(5))

        st.pyplot()

        st.subheader("Top 5 most frequent from each features pie chart")
    for col in categorical_features:
        plt.title(f"Pie char of {col}")
        df[col].value_counts(ascending=False).head(5).plot(kind='pie', autopct="%1.2f%%")

        st.pyplot()


## =========================== analysis ============================

elif side_bar_selected == 'Analysis':
    # ------------------------------------------
    st.subheader("Top 10 most frequent cart with interior designe")
    df[['Model', 'Leather_interior']].value_counts(ascending=False).head(10).plot(kind='bar')
    plt.xticks(rotation=25)
    st.pyplot()

    # ---------------------------------------------
    st.subheader("Leather Interior Type")
    plt.pie(df['Leather_interior'].value_counts(ascending=False).head(5), autopct="%1.1f%%")
    st.pyplot()

    ## -------------------------------------------------
    ## top 10 model cars with leather
    st.subheader("Top 10 cars with Leather interior design")
    top_10=df[df['Leather_interior'] == 'Yes'][['Model', 'Leather_interior', 'Price']].head(10)
    st.dataframe(top_10, hide_index=True)
else:
    st.title("Thanks a lot to visit on this Page")
    st.markdown("""<p style = color:#119730;font-size:20px;>My name is kamal. i am a data scienties, generative Ai and Machine learning Engineer</p>""",\
                 unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        image_path = "img/GitHub-Logo.png"
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
                # Create the HTML for the image with a link    
        html = f"""
        <a href="https://github.com/Kamaljit12/">
            <img src="data:image/png;base64,{encoded_image}" width="80">
        </a>
        """

        # Display the image with the link in Streamlit
        st.markdown(html, unsafe_allow_html=True)



        with col2:
            image_path = "img/linkedin-icon.jpg"
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
                    # Create the HTML for the image with a link    
            html = f"""
            <a href="https://shorturl.at/XdBUE">
                <img src="data:image/png;base64,{encoded_image}" width="50">
            </a>
            """

            # Display the image with the link in Streamlit
            st.markdown(html, unsafe_allow_html=True)



