# Symptify
Symptify is an interactive symptom diagnosis tool that predicts medical conditions using a trained Decision Tree Classifier. 
Users can select symptoms, view the predicted condition, and explore detailed drug information along with precautionary measures. 
This application is deployed on Posit Cloud, making it accessible and practical for real-world use.

## Features
- **Symptom Selection**: Users can choose from a comprehensive list of symptoms via an intuitive interface.
- **Condition Prediction**: Predicts medical conditions based on user inputs using a machine learning model.
- **Drug Information**: Provides detailed insights into related drugs, including usage and side effects.
- **Precautionary Measures**: Suggests preventive steps to manage or avoid worsening of the predicted condition.

## Steps to run the project
1. Clone the project project and navigate to the current directory
   ```bash
   git clone https://github.com/anirudh-hegde/symptify.git
   cd symptify
   ```
2. Create a virtual environment and install the requirements
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Execute the project
   ```bash
   streamlit run symp_app.py
   ```

## How to Use
1. Open the application on Posit Cloud.
2. Select the symptoms from the sidebar interface.
3. View the predicted medical condition displayed on the main screen.
4. Explore the related drug information and precautions provided.

## Deployment
Symptify is deployed on **Posit Cloud**, ensuring smooth and seamless access to users. This deployment leverages cloud scalability and 
reliability to handle multiple users efficiently.

Deployed web app link: [https://bit.ly/4guW19R](https://bit.ly/4guW19R)
