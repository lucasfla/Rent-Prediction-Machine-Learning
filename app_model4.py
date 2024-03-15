import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st
# pip install streamlit

apartments = pd.read_csv('apartments_for_rent_classified_10K.csv')
apartments = apartments[apartments['price_type'] == "Monthly"]
apartments = apartments.drop(columns=['currency', 'price_display', 'id', 'Unnamed: 0', 'fee', 'price_type'])
apartments['bedrooms'] = apartments['bedrooms'].astype(float)
apartments['square_feet'] = apartments['square_feet'].astype(float)

# dropping all rows with any na values
apartments = apartments.dropna(axis = 0)
apartments = apartments[apartments['price'] < 2700]
cityname_counts = apartments['cityname'].value_counts()
big_cities = cityname_counts[cityname_counts > 30].index
filtered_df = apartments[apartments['cityname'].isin(big_cities)]
apartments_mlr = filtered_df.copy()

# one-hot-encode state variables
one_hot_encoded = pd.get_dummies(apartments_mlr['state'])

# get a copy of the dataset for one-hot-encoding
apartments_mlr = filtered_df.copy().drop(columns=['bathrooms', 'bedrooms'])

# one-hot-encode state variables
one_hot_encoded = pd.get_dummies(apartments_mlr['state'])
# Concatenate apartments with the one-hot encoded variables
one_hot_encoded = one_hot_encoded.astype('int')
apartments_mlr = pd.concat([apartments_mlr, one_hot_encoded], axis=1).drop('state', axis = 1)

# one-hot-encode cityname variables
one_hot_encoded = pd.get_dummies(apartments_mlr['cityname'])
# Concatenate apartments with the one-hot encoded variables
one_hot_encoded = one_hot_encoded.astype('int')
apartments_mlr = pd.concat([apartments_mlr, one_hot_encoded], axis=1).drop('cityname', axis = 1)


# one-hot-encode has_photo variables
one_hot_encoded = pd.get_dummies(apartments_mlr['has_photo'])
# Concatenate apartments with the one-hot encoded variables
one_hot_encoded = one_hot_encoded.astype('int')
apartments_mlr = pd.concat([apartments_mlr, one_hot_encoded], axis=1).drop('has_photo', axis = 1)

# one-hot-encode pets_allowed variables
one_hot_encoded = pd.get_dummies(apartments_mlr['pets_allowed'])
# Concatenate apartments with the one-hot encoded variables
one_hot_encoded = one_hot_encoded.astype('int')
apartments_mlr = pd.concat([apartments_mlr, one_hot_encoded], axis=1).drop('pets_allowed', axis = 1)

# save a copy of processed data for decision tree model
apartments_dt = apartments_mlr.copy()

train, test = train_test_split(apartments_mlr, test_size=0.2, random_state=10)

# splitting up the predictor and prediction variables for both the training and testing data
X_train4, y_train = train.drop(columns=['price']), train['price']
X_test, y_test = test.drop(columns=['price']), test['price']

# Initialize LinearRegression model
model4 = LinearRegression()

# Fit the model to the training data
model4.fit(X_train4, y_train)

# Make predictions on the testing data
y_pred = model4.predict(X_test)
y_pred_train = model4.predict(X_train4)


param_grid = {
    'max_depth': [1, 3, 5, 7, 9, 10, 12, 15, 17, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4]
}
'''

'''

# Streamlit UI
st.title("Apartment Rent Price Prediction")

# Numeric inputs
square_feet = st.slider("Square Feet",
                        min_value=int(apartments['square_feet'].min()),
                        max_value=int(apartments['square_feet'].max()),
                        step=50)

has_photo0 = sorted(apartments['has_photo'].unique())
has_photo = st.selectbox("has_photo", has_photo0)
# Binary input
pets_allowed0 = sorted(apartments['pets_allowed'].unique())
pets_allowed = st.selectbox("Pets Allowed", pets_allowed0)
unique_states = sorted(apartments['state'].unique())
state = st.selectbox("State", unique_states)

# Instead of sorting the dictionary, we will create it and then sort the list within each state
cities_by_state = {state: sorted(apartments[apartments['state'] == state]['cityname'].unique().tolist())
                   for state in unique_states}

# Use state to get the list of cities for the selected state
city = st.selectbox("City", cities_by_state[state])

def predict_rent_price(square_feet, has_photo, pets_allowed, state, city, X_train_columns):
    # Initialize an empty DataFrame with expected structure
    input_data = pd.DataFrame(columns=X_train_columns)
    input_data.loc[0] = 0  # Start with all zeros

    # Set direct values
    input_data.at[0, 'square_feet'] = square_feet

    # One-hot encode 'state', 'cityname', 'has_photo', 'pets_allowed'
    if state in X_train_columns: input_data.at[0, state] = 1
    if city in X_train_columns: input_data.at[0, city] = 1
    if has_photo in X_train_columns: input_data.at[0, has_photo] = 1
    if pets_allowed in X_train_columns: input_data.at[0, pets_allowed] = 1

    # Ensure the input_data is in the same order as X_train_columns
    input_data = input_data.reindex(columns=X_train_columns)

    # Predict the price using the model
    prediction = model4.predict(input_data)
    return prediction[0]

# Predict button
if st.button('Predict'):
    # Create input data for the model
    prediction = predict_rent_price(square_feet, has_photo, pets_allowed, state,
                                    city, X_train4.columns)
    # Display prediction
    st.success(f"Estimated Rent Price: ${prediction:.2f}")

# py -m streamlit run app.py
