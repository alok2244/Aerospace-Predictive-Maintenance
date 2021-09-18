{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "43e00e3c-6d81-49cb-b234-5b43225d0a2c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running locally at: http://127.0.0.1:7867/\n",
      "To create a public link, set `share=True` in `launch()`.\n",
      "Interface loading below...\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"900\"\n",
       "            height=\"500\"\n",
       "            src=\"http://127.0.0.1:7867/\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.IFrame at 0x1755ede9160>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "(<Flask 'gradio.networking'>, 'http://127.0.0.1:7867/', None)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[31.0, 642.58, 1581.22, 1398.91, 554.42, 2388.08, 9056.4, 47.23, 521.79, 2388.06, 8130.11, 8.4024, 393.0, 38.81, 23.3552]\n",
      "[146.95628382]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import joblib\n",
    "import gradio as gr\n",
    "def predict_RUL_Binary(\n",
    "    Cycle,\n",
    "    SensorMeasure2,\n",
    "    SensorMeasure3,\n",
    "    SensorMeasure4,\n",
    "    SensorMeasure7,\n",
    "    SensorMeasure8,\n",
    "    SensorMeasure9,\n",
    "    SensorMeasure11,\n",
    "    SensorMeasure12, \n",
    "    SensorMeasure13,\n",
    "    SensorMeasure14, \n",
    "    SensorMeasure15, \n",
    "    SensorMeasure17,\n",
    "    SensorMeasure20,\n",
    "    SensorMeasure21):\n",
    "   \n",
    "    \n",
    "    input_data=[Cycle,\n",
    "    SensorMeasure2,\n",
    "    SensorMeasure3,\n",
    "    SensorMeasure4,\n",
    "    SensorMeasure7,\n",
    "    SensorMeasure8,\n",
    "    SensorMeasure9,\n",
    "    SensorMeasure11,\n",
    "    SensorMeasure12, \n",
    "    SensorMeasure13,\n",
    "    SensorMeasure14, \n",
    "    SensorMeasure15, \n",
    "    SensorMeasure17,\n",
    "    SensorMeasure20,\n",
    "    SensorMeasure21]\n",
    "    \n",
    "    print(input_data)\n",
    "    \n",
    "    scaler = joblib.load('scaler.sav')\n",
    "    \n",
    "    a = np.array(input_data)\n",
    "    a=np.reshape(a,(1, a.size))\n",
    "    a= scaler.transform(a)\n",
    "    \n",
    "    reg = joblib.load('clf_reg.sav')\n",
    "    \n",
    "    RUL=reg.predict(a)\n",
    "    print(RUL)\n",
    "    \n",
    "    _bin = joblib.load('clf_bin.sav')\n",
    "    \n",
    "    RUL_Binary=_bin.predict(a)\n",
    "    mapping=lambda x: \"Engine Is Okay\" if x==1 else \"Engine Is Not Okay\"\n",
    "    return mapping(RUL_Binary) ,int(RUL)\n",
    "    \n",
    "UIF=gr.Interface(predict_RUL_Binary,\n",
    "             [  gr.inputs.Number(label=\"Cycle\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure2\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure3\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure4\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure7\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure8\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure9\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure11\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure12\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure13\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure14\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure15\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure17\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure20\"), \n",
    "                gr.inputs.Number(label=\"SensorMeasure21\"), \n",
    "               \n",
    "            ],[\n",
    "                 gr.outputs.Label(num_top_classes=None, type=\"auto\", label=\"ENGINE CONDITION\"),\n",
    "                 gr.outputs.Label(num_top_classes=None, type=\"auto\", label=\"REMAINING USEFULL LIFE\")\n",
    "                 \n",
    "             ])\n",
    "    \n",
    "    \n",
    "UIF.launch()    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "926265ec-8b56-435b-a3ca-26f06c4ebab9",
   "metadata": {},
   "outputs": [],
   "source": [
    "[31.0, 642.58, 1581.22, 1398.91, 554.42, 2388.08, 9056.4, 47.23, 521.79, 2388.06, 8130.11, 8.4024, 393.0, 38.81, 23.3552]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
