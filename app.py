import streamlit as st
import numpy as np
import pickle

with open("Best_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
with open("column_transformer.pkl", "rb") as file:
    clt = pickle.load(file)
with open("encoder.pkl", "rb") as file:
    enc = pickle.load(file)
cols = ['Buying_Cost',
        'Maintainance_Cost',
        'Number_of_doors',
        'Number_of_Passenger',
        'Luggage_Space',
        'Safety_Features']
val = [['high', 'low', 'med', 'vhigh'],
       ['high', 'low', 'med', 'vhigh'],
       ['2', '3', '4', '5more'],
       ['2', '4', 'more'],
       ['big', 'med', 'small'],
       ['high', 'low', 'med']]

def main():
    st.header('Car Selling recommendations !!')
    st.write("We have started a Car Selling business and are giving recommendations to people for buying cars.")
    inp = []
    for id, v in enumerate(val):
        i = st.selectbox(cols[id], v).strip()
        inp.append(i)

    sub_btn = st.button("Submit")
    if sub_btn:
        pred = loaded_model.predict(clt.transform([inp]))
        out = enc.inverse_transform(pred)[0]
        if out == "Bad_deal":
            st.write("It is not going to be good deal !! 😔")
        else:    
            st.write("It is going to be a great deal !! 😃")

if __name__ == '__main__':
    main()