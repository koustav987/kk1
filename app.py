from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store model and data info
model = None
feature_names = None
categorical_cols = None
numerical_cols = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def clean_data(df):
    """Clean the uploaded data"""
    print("Initial shape:", df.shape)
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Fix non-numeric rainfall
    def isStr(obj):
        try:
            float(obj)
            return False
        except:
            return True
    
    if 'Annual_Rainfall' in df.columns:
        non_numeric = df['Annual_Rainfall'].apply(isStr)
        df = df[~non_numeric]
        df['Annual_Rainfall'] = pd.to_numeric(df['Annual_Rainfall'], errors='coerce')
    
    df = df.dropna()
    print("After cleaning:", df.shape)
    return df

def train_model(df):
    """Train the crop yield prediction model"""
    global model, feature_names, categorical_cols, numerical_cols
    
    target = 'Yield'
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify categorical & numerical columns
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols),
        ('scale', StandardScaler(), numerical_cols)
    ])
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, mod in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', mod)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {'R2': r2, 'RMSE': rmse, 'model': pipeline}
    
    # Select best model
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    model = results[best_model_name]['model']
    
    # Save model
    joblib.dump(model, 'crop_yield_model.pkl')
    
    return results, best_model_name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process data
            df = pd.read_csv(filepath)
            df = clean_data(df)
            
            # Train model
            results, best_model_name = train_model(df)
            
            flash(f'Model trained successfully! Best model: {best_model_name}')
            return render_template('results.html', results=results, best_model=best_model_name)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('home'))
    
    flash('Invalid file format. Please upload a CSV file.')
    return redirect(url_for('home'))

@app.route('/predict')
def predict_page():
    if model is None:
        flash('Please train a model first by uploading data.')
        return redirect(url_for('home'))
    
    return render_template('predict.html', 
                         categorical_cols=categorical_cols, 
                         numerical_cols=numerical_cols)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    try:
        # Get form data
        input_data = {}
        for col in feature_names:
            if col in categorical_cols:
                input_data[col] = request.form.get(col, '')
            else:
                input_data[col] = float(request.form.get(col, 0))
        
        # Create DataFrame and predict
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        
        return render_template('prediction_result.html', 
                             prediction=round(prediction, 2),
                             input_data=input_data)
    
    except Exception as e:
        flash(f'Error making prediction: {str(e)}')
        return redirect(url_for('predict_page'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return jsonify({'Predicted_Yield': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load existing model if available
    if os.path.exists('crop_yield_model.pkl'):
        try:
            model = joblib.load('crop_yield_model.pkl')
            print("Existing model loaded successfully!")
        except:
            print("Could not load existing model")
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))