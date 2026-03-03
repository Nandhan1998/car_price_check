# from flask import Flask, render_template, request
# import pandas as pd
# import pickle

# app = Flask(__name__)

# # Load the model and columns exactly as saved in your notebook
# with open('ridge_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open('model_columns.pkl', 'rb') as f:
#     model_columns = pickle.load(f)


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Retrieve data from index.html form fields
#     form_data = {
#         'symboling': int(request.form.get('symboling')),
#         'make': int(request.form.get('make')),
#         'fuel-type': int(request.form.get('fuel-type')),
#         'aspiration': int(request.form.get('aspiration')),
#         'num-of-doors': int(request.form.get('num-of-doors')),
#         'body-style': int(request.form.get('body-style')),
#         'drive-wheels': int(request.form.get('drive-wheels')),
#         'engine-location': int(request.form.get('engine-location')),
#         'wheel-base': float(request.form.get('wheel-base')),
#         'length': float(request.form.get('length')),
#         'width': float(request.form.get('width')),
#         'height': float(request.form.get('height')),
#         'engine-type': int(request.form.get('engine-type')),
#         'num-of-cylinders': int(request.form.get('num-of-cylinders')),
#         'engine-size': int(request.form.get('engine-size')),
#         'fuel-system': int(request.form.get('fuel-system')),
#         'bore': float(request.form.get('bore')),
#         'stroke': float(request.form.get('stroke')),
#         'compression-ratio': float(request.form.get('compression-ratio')),
#         'highway-mpg': int(request.form.get('highway-mpg'))
#     }

#     # Create DataFrame and ensure the column order matches training data
#     df_new = pd.DataFrame([form_data])
#     df_new = df_new[model_columns]

#     # Predict the price
#     prediction = model.predict(df_new)[0]

#     return render_template('index.html', prediction_text=f'Estimated Price: ${prediction:,.2f}')


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'ridge_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'model_columns.pkl'), 'rb') as f:
    model_columns = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all 20 inputs from the form
        data = {
            'symboling': float(request.form.get('symboling')),
            'make': float(request.form.get('make')),
            'fuel-type': float(request.form.get('fuel-type')),
            'aspiration': float(request.form.get('aspiration')),
            'num-of-doors': float(request.form.get('num-of-doors')),
            'body-style': float(request.form.get('body-style')),
            'drive-wheels': float(request.form.get('drive-wheels')),
            'engine-location': float(request.form.get('engine-location')),
            'wheel-base': float(request.form.get('wheel-base')),
            'length': float(request.form.get('length')),
            'width': float(request.form.get('width')),
            'height': float(request.form.get('height')),
            'engine-type': float(request.form.get('engine-type')),
            'num-of-cylinders': float(request.form.get('num-of-cylinders')),
            'engine-size': float(request.form.get('engine-size')),
            'fuel-system': float(request.form.get('fuel-system')),
            'bore': float(request.form.get('bore')),
            'stroke': float(request.form.get('stroke')),
            'compression-ratio': float(request.form.get('compression-ratio')),
            'highway-mpg': float(request.form.get('highway-mpg'))
        }

        # Convert to DataFrame and align columns to match training set
        df_new = pd.DataFrame([data])
        df_new = df_new[model_columns]

        # Predict
        prediction = model.predict(df_new)[0]

        # Display the result (formatted as currency)
        return render_template('index.html',
                               prediction_text=f'Estimated Price: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in calculation: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
