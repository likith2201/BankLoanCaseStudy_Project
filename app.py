import streamlit as st
from PIL import Image
%pip  install joblib
import joblib
import pandas as pd

# Load the trained model


model1 = joblib.load('rf_model1.joblib')


model2 = joblib.load('XGBModel1.joblib')

ownership_mapping = {"RENT": 1, "OWN": 2, "MORTGAGE": 3, "OTHER": 4}

purpose_mapping = {
    "credit_card": 1, "car": 2, "small_business": 3, "other": 4, "wedding": 5,
    "debt_consolidation": 6, "home_improvement": 7, "major_purchase": 8, "medical": 9,
    "moving": 10, "vacation": 11, "house": 12, "renewable_energy": 13, "educational": 14,
}

state_mapping = {
    "AZ": 1, "GA": 2, "IL": 3, "CA": 4, "NC": 5, "TX": 6, "VA": 7, "MO": 8,
    "CT": 9, "UT": 10, "FL": 11, "PA": 12, "MN": 13, "NY": 14, "NJ": 15,
    "OR": 16, "KY": 17, "OH": 18, "SC": 19, "RI": 20, "LA": 21, "MA": 22,
    "WA": 23, "WI": 24, "AL": 25, "NV": 26, "AK": 27, "CO": 28, "MD": 29,
    "WV": 30, "VT": 31, "MI": 32, "DC": 33, "SD": 34, "NH": 35, "AR": 36,
    "NM": 37, "KS": 38, "HI": 39, "OK": 40, "MT": 41, "WY": 42, "DE": 43,
    "MS": 44, "TN": 45, "IA": 46, "NE": 47, "ID": 48, "IN": 49
}


def run():
    img1 = Image.open('bank.png')
    img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("Bank Loan Prediction using Machine Learning")

    
    loan_amnt = st.slider('Loan Amount', min_value=1000, max_value=25000)

    installment = st.slider('Installment Amount', min_value=10, max_value=1000)

    annual_inc = st.text_input('Annual  income in K $')

    dti = st.slider('Debt to Income ratio', min_value=0, max_value=30)


    
    home_ownership_display = ('RENT', 'OWN', 'MORTGAGE', 'OTHER')
    home_options = list(range(len(home_ownership_display)))
    home_ownership = st.selectbox("Home Status",home_options, format_func=lambda x: home_ownership_display[x])

    
    purp_display = ('credit_card', 'car', 'small_business', 'other', 'wedding', 'debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'moving', 'vacation', 'house', 'renewable_energy', 'educational')
    purp_options = list(range(len(purp_display)))
    purpose = st.selectbox("Loan Purpose", purp_options, format_func=lambda x: purp_display[x])

    
    state_display = ('AZ', 'GA', 'IL', 'CA', 'NC', 'TX', 'VA', 'MO', 'CT', 'UT', 'FL', 'PA', 'MN', 'NY', 'NJ', 'OR', 'KY', 'OH', 'SC', 'RI', 'LA', 'MA', 'WA', 'WI', 'AL', 'NV', 'AK', 'CO', 'MD', 'WV', 'VT', 'MI', 'DC', 'SD', 'NH', 'AR', 'NM', 'KS', 'HI', 'OK', 'MT', 'WY', 'DE', 'MS', 'TN', 'IA', 'NE', 'ID', 'IN')
    state_options = list(range(len(state_display)))
    state = st.selectbox("State Code", state_options, format_func=lambda x: state_display[x])


    
    lp_display = ('36', '60')
    lp_options = list(range(len(lp_display)))
    loanPeriod = st.selectbox("Loan Period",  lp_options, format_func=lambda x: lp_display[x])

    int_rate = st.text_input('Expected  Interest Rate %','Enter the value between 0.1 and 30')


    if st.button("Submit"):
        
        # Create a DataFrame with the raw inputs
            features_raw = pd.DataFrame({
                'loan_amnt': [loan_amnt],
                'installment': [installment],
                'annual_inc': [annual_inc],
                'dti': [dti],
                'home_ownership': [home_ownership_display[int(home_ownership)]],  # Convert index to label
                'purpose': [purp_display[int(purpose)]],  # Convert index to label
                'loanPeriod': [lp_display[int(loanPeriod)]],  # Assuming this is correct
                'state': [state_display[int(state)]],  # Convert index to label
            })
    
         # Apply mappings to convert labels to their numerical representations
            features_raw['home_ownership'] = features_raw['home_ownership'].map(ownership_mapping)
            features_raw['purpose'] = features_raw['purpose'].map(purpose_mapping)
            features_raw['state'] = features_raw['state'].map(state_mapping)


            features_raw_l = pd.DataFrame({
                'loan_amnt': [loan_amnt],
                'installment': [installment],
                'annual_inc': [annual_inc],
                'dti': [dti],
                'home_ownership': [home_ownership_display[int(home_ownership)]],  # Convert index to label
                'purpose': [purp_display[int(purpose)]],  # Convert index to label
                'loanPeriod': [lp_display[int(loanPeriod)]],  # Assuming this is correct
                'state': [state_display[int(state)]],  # Convert index to label
                'int_rate%': [float(int_rate)]   # Add interest rate as a feature
            })
    
         # Apply mappings to convert labels to their numerical representations
            features_raw_l['home_ownership'] = features_raw_l['home_ownership'].map(ownership_mapping)
            features_raw_l['purpose'] = features_raw_l['purpose'].map(purpose_mapping)
            features_raw_l['state'] = features_raw_l['state'].map(state_mapping)

                
            
            prediction1 = model1.predict(features_raw_l)

            # Get the prediction probabilities
            prediction1_prob = model1.predict_proba(features_raw_l)

            # Create a DataFrame with the probabilities and custom column names
            prob_df = pd.DataFrame(prediction1_prob, columns=["Reject", "Accept"])

            # Check the prediction and set the message accordingly with the new condition
            if prediction1[0] == 1 and prediction1_prob[0][1] >= 0.75:
                prediction_message = "Accept"
            else:
                prediction_message = "Reject"

            # Use the prediction_message in your output
            st.write(f'Predicted loan status: {prediction_message}')

            # Display the DataFrame using Streamlit
            st.write('Probability of prediction:')
            st.dataframe(prob_df)

            # If the prediction is to accept and the accept probability is >= 0.75, proceed to predict interest rate
            if prediction1[0] == 1 and prediction1_prob[0][1] >= 0.70:
                prediction2 = model2.predict(features_raw)
                st.write(f'Predicted interest rate: {prediction2[0]:,.2f}%')

run()