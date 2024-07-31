from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained models and their accuracies
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
decision_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
model_accuracies = pickle.load(open('model_accuracies.pkl', 'rb'))

# Load the pesticide data
pesticide_data = pd.read_csv('pesticide_management.csv')

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_values = [
        float(request.form['nitrogen']),
        float(request.form['phosphorous']),
        float(request.form['potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    # Make predictions using all models
    rf_recommended_crop = random_forest_model.predict([input_values])[0]
    dt_recommended_crop = decision_tree_model.predict([input_values])[0]
    svm_recommended_crop = svm_model.predict([input_values])[0]
    knn_recommended_crop = knn_model.predict([input_values])[0]
    logistic_recommended_crop = logistic_model.predict([input_values])[0]

    # Determine the best recommendation based on accuracy
    accuracies = {
        'random_forest': model_accuracies['random_forest'],
        'decision_tree': model_accuracies['decision_tree'],
        'svm': model_accuracies['svm'],
        'knn': model_accuracies['knn'],
        'logistic': model_accuracies['logistic']
    }
    
    best_model = max(accuracies, key=accuracies.get)
    
    best_crop = {
        'random_forest': rf_recommended_crop,
        'decision_tree': dt_recommended_crop,
        'svm': svm_recommended_crop,
        'knn': knn_recommended_crop,
        'logistic': logistic_recommended_crop
    }[best_model]

    return render_template('result.html', 
                           rf_crop=rf_recommended_crop, 
                           dt_crop=dt_recommended_crop, 
                           svm_crop=svm_recommended_crop, 
                           knn_crop=knn_recommended_crop,
                           logistic_crop=logistic_recommended_crop,
                           rf_accuracy=model_accuracies['random_forest'], 
                           dt_accuracy=model_accuracies['decision_tree'], 
                           svm_accuracy=model_accuracies['svm'],
                           knn_accuracy=model_accuracies['knn'],
                           logistic_accuracy=model_accuracies['logistic'],
                           best_crop=best_crop,
                           best_model=best_model.capitalize())

@app.route('/pesticide_management')
def pesticide_management():
    return render_template('pesticide.html', crop=None, diseases=None)

@app.route('/pesticide', methods=['GET', 'POST'])
def pesticide():
    if request.method == 'POST':
        crop = request.form['crop'].lower()
        crop_info = pesticide_data[pesticide_data['Recommended_Crop'].str.lower() == crop]
        
        if not crop_info.empty:
            diseases = crop_info[['Potential_Disease', 'Disease_Solution', 'Recommended_Pesticide']].to_dict('records')
        else:
            diseases = None

        return render_template('pesticide.html', crop=crop, diseases=diseases)
    else:
        return render_template('pesticide.html', crop=None, diseases=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
