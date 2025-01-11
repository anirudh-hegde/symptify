import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Symptify", page_icon="⚕️")

with open('style.css') as f:
	st.markdown(f'''<style>{f.read()}</style>''', unsafe_allow_html=True)

symptoms_df = pd.read_csv('dataset/symptoms_dataset.csv')
drugs_df = pd.read_csv('dataset/drugs_side_effects.csv')
precautions_df = pd.read_csv('dataset/disease_precaution.csv')

symptoms_df['TYPE'] = symptoms_df['TYPE'].replace(
	{'COLD': 'Colds & Flu', 'FLU': 'Colds & Flu', 'COVID': 'Covid 19', 'ALLERGY': 'Allergies'})

X = symptoms_df.drop('TYPE', axis=1)
y = symptoms_df['TYPE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

param_grid_rf = {
    # "n_estimators": [50,100,200],
    # "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=model, param_grid=param_grid_rf, cv=5, scoring='accuracy')
# n_jobs=-1, verbose=2)
# X_train=X_train.values
# model.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)

rf_model = grid_search_rf.best_estimator_
st.title("⚕️Symptom Diagnosis and Drug Information Tool")
st.markdown("### Check your symptoms and get a diagnosis along with drug info!")

st.sidebar.header("Select Your Symptoms")

symptoms = X.columns
user_input = []

for symptom in symptoms:
	# st.sidebar.checkbox(symptom)
	user_input.append(st.sidebar.checkbox(symptom.title().replace("_","")))

# user_input = np.array(user_input).reshape(1, -1)
user_input = np.array(user_input).reshape(1, -1)
# print(user_input)
if user_input.any():
	prediction = rf_model.predict(user_input)[0]
	st.markdown(f"## Predicted Condition: **:red[{prediction}]**")

	st.markdown("### Precaution Information:")
	precaution_info = precautions_df[precautions_df['Disease'].str.contains(prediction, case=False, na=False)]

	if not precaution_info.empty:
		st.markdown("#### Precautions:")
		for i in range(1, 5):
			precaution_column = f'Precaution_{i}'
			if precaution_column in precaution_info.columns:
				precaution = str(precaution_info.iloc[0][precaution_column]).capitalize()
				# if precaution == "Nan":
					# continue
				st.write(f"- {precaution}")
	else:
		st.markdown("No precaution information available for this condition.")

	drug_info = drugs_df[drugs_df['medical_condition'].str.contains(prediction, case=False, na=False)]

	if not drug_info.empty:
		st.markdown("### Related Drugs:")

		selected_drug_name = st.selectbox("Select a drug to see more details:", drug_info['drug_name'].unique())

		if st.button("Info", key="btn"):
			# if selected_drug_name:
			st.markdown(f"## Drug Information: **{selected_drug_name}**")
			selected_drug = drug_info[drug_info['drug_name'] == selected_drug_name].iloc[0]
			for col in drug_info.columns:
				if col != 'drug_name':
					st.write(f"**{col.replace('_', ' ').title()}:** {selected_drug[col]}")
	else:
		st.markdown("No drug information available for this condition.")
else:
	st.error("Choose at least one symptom to receive your condition prediction.")

st.text("")
st.markdown("""---""")
st.markdown("### Usage")
st.write("""
1. Choose your symptoms from the sidebar.
   The app will display the predicted condition and related medications.
   Pick a drug to see its detailed information.

2. Select your symptoms from the sidebar.
   The predicted condition and associated drug info will update automatically.
   Choose a drug to get more details about it.

3. Pick your symptoms from the sidebar.
   Your diagnosis and related medications will be displayed.
   Click on a drug name to learn more about it.

4. Select your symptoms using the sidebar.
   Your condition and relevant drug suggestions will appear.
   Select a drug to view detailed information.

5. Check your symptoms from the sidebar.
   The system will show the predicted condition and medication options.
   Select a drug for more details about its usage and effects.
""")

st.markdown("""---""")

st.write(""" ### Disclaimer - Please note, this app is **NOT** a replacement for professional medical advice, diagnosis, or treatment.
Always seek guidance from your doctor or a certified healthcare provider with any health-related questions. -
The content is intended solely for educational and informational purposes. """)

st.markdown("""---""")

st.markdown("### About the Application")

st.write("""
This application predicts common health conditions based on the symptoms you choose and provides information about related medications.
    The model has been trained on symptom and condition datasets, linked with drug information to offer thorough insights.""")

st.markdown("""---""")

st.markdown("### Code Repository")
st.write("""You can find the complete code for this application in my GitHub repository [here](https://github.com/anirudh-hegde/).
""")
<<<<<<< HEAD


from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", scores.mean()) # Cross-Validation Accuracy: 0.94
