from flask import Flask, render_template, request 
import numpy as np
import pandas as pd 
import pickle 
from rapidfuzz import process

# load dataset
# load all the dataframes togather in one cell else it wont allow the helper function to run
Symptom_disease = pd.read_csv('Datasets/symtoms_df.csv')
precautions = pd.read_csv('Datasets/precautions_df.csv')
description = pd.read_csv('Datasets/description.csv')
medications = pd.read_csv('Datasets/medications.csv')
diets = pd.read_csv('Datasets/diets.csv')
workout = pd.read_csv('Datasets/workout_df.csv')

# load models 
# load Model rb means read binary
svc = pickle.load(open('Datasets/models/Trained_model.pkl','rb'))

app = Flask(__name__)

#  we copied from our own collab 
# Helper Function jo disease name receive karega and based on that it will tell description,precaution,diets,medications,workout



def helper_Function(disease):
  # get description
  descr = description[description['Disease'] == disease]['Description']
  descr = " ".join([w for w in descr])

  # get Precautions
  prec = precautions[precautions['Disease'] == disease ][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
  prec = [col for col in prec.values]

  # get Diet
  diet = diets[diets['Disease'] == disease ]['Diet']
  diet = [col for col in diet.values]

  # get Medications
  med = medications[medications['Disease'] == disease ]['Medication']
  med = [col for col in med.values]

  # get workout
  wout = workout[workout['disease'] == disease ]['workout']
  wout = [col for col in wout.values]

  # returns all the calculated values
  return descr, prec, diet, med, wout


# model prediction function

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# it can take any number of symptoms and tells name of disease

def get_disease_recommendation(patients_symptoms):
  input_vector = np.zeros(len(symptoms_dict))  # shape should be always perfect else many errors

  for items in patients_symptoms:
    input_vector[symptoms_dict[items]] = 1                 # so pehle all zeros the abhi jo symptoms hai waha 1 enter hoga so humara model predict kar sake
    return  diseases_list[svc.predict([input_vector])[0]]  # so yaha pr humara code 15 predict karega tho 15 ka value fungal infection return hoga means key ki value
                                                           # yaha pr jo array hai woh 2D mein bhi convert hua hai kyuki input_vector [] ke under hai

# this function acts like auto correct 

# Your dictionary of symptoms

# user symptoms are passed here as the input text and this function corrects the user input as per the model we trained 
def correct_symptom(input_text, symptoms_dict, threshold=80):  
    """
    Correct user input based on the closest match from the dictionary keys.
    Args:
    - input_text (str): The user's input text (e.g., "small dint in nail").
    - symptoms_dict (dict): The dictionary of valid symptoms.
    - threshold (int): Similarity threshold (default=80).
    Returns:
    - str: The corrected symptom key from the dictionary or a "No match" message.
    """

    # Perform fuzzy matching to find the best match in the dictionary keys
    best_match = process.extractOne(input_text, symptoms_dict.keys())
    
    # If a match is found and meets the threshold, return the corrected key
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    else:
        return "No suitable match found"


# creating the routes 
@app.route('/')
def index():
    return render_template('index.html')

# here we will implement all our prediction logic 
@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [symptoms.strip() for symptoms in symptoms.split(',')]

        # removing extra characters if any 
        user_symptoms = [ sym.strip("[]' ") for sym in user_symptoms] # converting to list

        # created a list to store the corrected Symptoms 
        corrected_symptoms = [] 

         # Correct each symptom using `correct_symptom` so user input is convert as per our requirement in symptoms_dict
        for symptom in user_symptoms:
            corrected = correct_symptom(symptom, symptoms_dict)
            if corrected:
                corrected_symptoms.append(corrected)

        if corrected_symptoms:
            predicted_disease = get_disease_recommendation(corrected_symptoms)

            # printing the results
            descr, prec, diet, med, wout = helper_Function(predicted_disease)
            print(corrected_symptoms)
            print(predicted_disease)

        # just because precautions,medications and diet are entirely printing in list 
        myprec = []
        mymed = []
        mydiet = []
        for i in prec[0]:
           myprec.append(i)

        # for i in med[0]:
        #    mymed.append(i)

        # for i in diet[0]:
        #    mydiet.append(i)      

    return render_template('index.html', predicted_disease = predicted_disease, disease_desc = descr, disease_prec =myprec, disease_diet =diet, disease_medication = med, disease_workout = wout)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')


# python main 
if __name__ == "__main__":
    app.run(debug = True)
    