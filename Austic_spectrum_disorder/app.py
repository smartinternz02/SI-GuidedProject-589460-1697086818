from flask import Flask,render_template,url_for,request,send_from_directory
import joblib
app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # print(model.predict([[1,1,1,1,0,0,1,1,0,0,26,6,0,0,0]]))
    A1_score = request.form['A1_score']
    A2_Score = request.form['A2_score']
    A3_Score = request.form['A3_score']
    A4_Score = request.form['A4_score']
    A5_Score = request.form['A5_score']
    A6_Score = request.form['A6_score']
    A7_Score = request.form['A7_score']
    A8_Score = request.form['A8_score']
    A9_Score = request.form['A9_score']
    A10_score = request.form['A10_score']
    age = request.form['age']
    result = request.form['result']
    m = request.form['m']
    Had_jaundice_yes = request.form['Had_jaundice_yes']
    Rel_had_yes = request.form['Rel_had_yes']
    X = [[int(A1_score), int(A2_Score), int(A3_Score),int(A4_Score), int(A5_Score), int(A6_Score),int(A7_Score), int(A8_Score), int(A9_Score),int(A10_score),int(age),int(result),int(m),int(Had_jaundice_yes),int(Rel_had_yes)]]
    prediction = model.predict(X)
    if prediction==False:
       prediction=0.0
    else:
        prediction=True
    return render_template('final.html', prediction='DETECTION: {}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
