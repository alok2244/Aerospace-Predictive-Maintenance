from flask import Flask,render_template,request
import numpy as np
import joblib
import pickle
import os
print('here3')
app=Flask(__name__)
print('here4')


APP_ROOT = os.path.dirname(os.path.abspath(__file__))     
# set file directory path

scaler_path = os.path.join(APP_ROOT, "./pickel/scaler.pkl")  
# set path to the scaler

clf_path = os.path.join(APP_ROOT, "./pickel/clf_bin.pkl") 
# set path to the clf

reg_path = os.path.join(APP_ROOT, "./pickel/clf_reg.pkl") 
# set path to the reg
    
with open(scaler_path, 'rb') as handle:
    scaler = pickle.load(handle)

with open(clf_path, 'rb') as handle:
    _bin = pickle.load(handle)
    
with open(reg_path, 'rb') as handle:
    reg = pickle.load(handle)
   

@app.route("/")
def home():
    print('here1')
   # return tf_idf,log_reg,naive_bayes}
    return render_template("main.html",output=False)

@app.route('/mainResult',methods=['POST'])
def prediction():
     input_list_name=['Cycle',
                      'SensorMeasure2',
                      'SensorMeasure3',
                      'SensorMeasure4',
                      'SensorMeasure7',
                      'SensorMeasure8',
                      'SensorMeasure9',
                      'SensorMeasure11',
                      'SensorMeasure12', 
                      'SensorMeasure13',
                      'SensorMeasure14', 
                      'SensorMeasure15', 
                      'SensorMeasure17',
                      'SensorMeasure20',
                      'SensorMeasure21']
     
     input_list_value=[]
     if request.method == 'POST':
         for i,name in enumerate(input_list_name):
            input_list_value.append(float(request.form[name]))
     
     print(input_list_value)
     print(type(input_list_value))
     
     
    
     a = np.array(input_list_value)
     a=np.reshape(a,(1, a.size))
     a= scaler.transform(a)
    
     
     
     RUL=reg.predict(a)
     print(RUL)
    
     
    
     RUL_Binary=_bin.predict(a)
     mapping=lambda x: "Engine Is Okay" if x==1 else "Engine Is Not Okay"
     engine_health=mapping(RUL_Binary)
     
     return render_template('main.html',output=True,RemainingUL=RUL,condition=engine_health)

if __name__ == '__main__':
    print('here2')
    app.run()
