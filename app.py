import numpy as np
import joblib
import gradio as gr
def predict_RUL_Binary(
    Cycle,
    SensorMeasure2,
    SensorMeasure3,
    SensorMeasure4,
    SensorMeasure7,
    SensorMeasure8,
    SensorMeasure9,
    SensorMeasure11,
    SensorMeasure12, 
    SensorMeasure13,
    SensorMeasure14, 
    SensorMeasure15, 
    SensorMeasure17,
    SensorMeasure20,
    SensorMeasure21):
   
    
    input_data=[Cycle,
    SensorMeasure2,
    SensorMeasure3,
    SensorMeasure4,
    SensorMeasure7,
    SensorMeasure8,
    SensorMeasure9,
    SensorMeasure11,
    SensorMeasure12, 
    SensorMeasure13,
    SensorMeasure14, 
    SensorMeasure15, 
    SensorMeasure17,
    SensorMeasure20,
    SensorMeasure21]
    
    print(input_data)
    
    scaler = joblib.load('scaler.sav')
    
    a = np.array(input_data)
    a=np.reshape(a,(1, a.size))
    a= scaler.transform(a)
    
    reg = joblib.load('clf_reg.sav')
    
    RUL=reg.predict(a)
    print(RUL)
    
    _bin = joblib.load('clf_bin.sav')
    
    RUL_Binary=_bin.predict(a)
    mapping=lambda x: "Engine Is Okay" if x==1 else "Engine Is Not Okay"
    return mapping(RUL_Binary) ,int(RUL)
    
UIF=gr.Interface(predict_RUL_Binary,
             [  gr.inputs.Number(label="Cycle"), 
                gr.inputs.Number(label="SensorMeasure2"), 
                gr.inputs.Number(label="SensorMeasure3"), 
                gr.inputs.Number(label="SensorMeasure4"), 
                gr.inputs.Number(label="SensorMeasure7"), 
                gr.inputs.Number(label="SensorMeasure8"), 
                gr.inputs.Number(label="SensorMeasure9"), 
                gr.inputs.Number(label="SensorMeasure11"), 
                gr.inputs.Number(label="SensorMeasure12"), 
                gr.inputs.Number(label="SensorMeasure13"), 
                gr.inputs.Number(label="SensorMeasure14"), 
                gr.inputs.Number(label="SensorMeasure15"), 
                gr.inputs.Number(label="SensorMeasure17"), 
                gr.inputs.Number(label="SensorMeasure20"), 
                gr.inputs.Number(label="SensorMeasure21"), 
               
            ],[
                 gr.outputs.Label(num_top_classes=None, type="auto", label="ENGINE CONDITION"),
                 gr.outputs.Label(num_top_classes=None, type="auto", label="REMAINING USEFULL LIFE")
                 
             ])
    
    
UIF.launch(share=True)    