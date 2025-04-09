from flask import Flask, request,render_template
import pickle
import numpy as np

app = Flask(__name__)

# Mapping from numerical prediction to severity label
severity_mapping = {
    0: 'Extremely Severe',
    1: 'Severe',
    2: 'Moderate',
    3: 'Mild',
    4: 'Normal'
}

# Define the 42 DASS questions with their text and subscale assignment
questions = {
    "Q1":  {"text": "I found myself getting upset by quite trivial things.", "scale": "stress"},
    "Q2":  {"text": "I was aware of dryness of my mouth.", "scale": "anxiety"},
    "Q3":  {"text": "I couldn't seem to experience any positive feeling at all.", "scale": "depression"},
    "Q4":  {"text": "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion).", "scale": "anxiety"},
    "Q5":  {"text": "I just couldn't seem to get going.", "scale": "depression"},
    "Q6":  {"text": "I tended to over-react to situations.", "scale": "stress"},
    "Q7":  {"text": "I had a feeling of shakiness (e.g., legs going to give way).", "scale": "anxiety"},
    "Q8":  {"text": "I found it difficult to relax.", "scale": "stress"},
    "Q9":  {"text": "I found myself in situations that made me so anxious I was most relieved when they ended.", "scale": "anxiety"},
    "Q10": {"text": "I felt that I had nothing to look forward to.", "scale": "depression"},
    "Q11": {"text": "I found myself getting upset rather easily.", "scale": "stress"},
    "Q12": {"text": "I felt that I was using a lot of nervous energy.", "scale": "stress"},
    "Q13": {"text": "I felt sad and depressed.", "scale": "depression"},
    "Q14": {"text": "I found myself getting impatient when I was delayed in any way (e.g., elevators, traffic lights, being kept waiting).", "scale": "stress"},
    "Q15": {"text": "I had a feeling of faintness.", "scale": "anxiety"},
    "Q16": {"text": "I felt that I had lost interest in just about everything.", "scale": "depression"},
    "Q17": {"text": "I felt I wasn't worth much as a person.", "scale": "depression"},
    "Q18": {"text": "I felt that I was rather touchy.", "scale": "stress"},
    "Q19": {"text": "I perspired noticeably (e.g., hands sweaty) in the absence of high temperatures or physical exertion.", "scale": "anxiety"},
    "Q20": {"text": "I felt scared without any good reason.", "scale": "anxiety"},
    "Q21": {"text": "I felt that life wasn't worthwhile.", "scale": "depression"},
    "Q22": {"text": "I found it hard to wind down.", "scale": "stress"},
    "Q23": {"text": "I had difficulty in swallowing.", "scale": "anxiety"},
    "Q24": {"text": "I couldn't seem to get any enjoyment out of the things I did.", "scale": "anxiety"},
    "Q25": {"text": "I was aware of the action of my heart in the absence of physical exertion (e.g., sense of heart rate increase, heart missing a beat).", "scale": "anxiety"},
    "Q26": {"text": "I felt down-hearted and blue.", "scale": "depression"},
    "Q27": {"text": "I found that I was very irritable.", "scale": "stress"},
    "Q28": {"text": "I felt I was close to panic.", "scale": "stress"},
    "Q29": {"text": "I found it hard to calm down after something upset me.", "scale": "anxiety"},
    "Q30": {"text": "I feared that I would be \"thrown\" by some trivial but unfamiliar task.", "scale": "anxiety"},
    "Q31": {"text": "I was unable to become enthusiastic about anything.", "scale": "depression"},
    "Q32": {"text": "I found it difficult to tolerate interruptions to what I was doing.", "scale": "stress"},
    "Q33": {"text": "I was in a state of nervous tension.", "scale": "stress"},
    "Q34": {"text": "I felt I was pretty worthless.", "scale": "stress"},
    "Q35": {"text": "I was intolerant of anything that kept me from getting on with what I was doing.", "scale": "stress"},
    "Q36": {"text": "I felt terrified.", "scale": "depression"},
    "Q37": {"text": "I could see nothing in the future to be hopeful about.", "scale": "depression"},
    "Q38": {"text": "I felt that life was meaningless.", "scale": "depression"},
    "Q39": {"text": "I found myself getting agitated.", "scale": "depression"},
    "Q40": {"text": "I was worried about situations in which I might panic and make a fool of myself.", "scale": "anxiety"},
    "Q41": {"text": "I experienced trembling (e.g., in the hands).", "scale": "anxiety"},
    "Q42": {"text": "I found it difficult to work up the initiative to do things.", "scale": "depression"}
}

# For each subscale, define the order of questions (must match the order used during model training)
depression_questions = ["Q3", "Q5", "Q10", "Q13", "Q16", "Q17", "Q21", "Q26", "Q31", "Q36", "Q37", "Q38", "Q39", "Q42"]
anxiety_questions    = ["Q2", "Q4", "Q7", "Q9", "Q15", "Q19", "Q20", "Q23", "Q24", "Q25", "Q29", "Q30", "Q40", "Q41"]
stress_questions     = ["Q1", "Q6", "Q8", "Q11", "Q12", "Q14", "Q18", "Q22", "Q27", "Q28", "Q32", "Q33", "Q34", "Q35"]

with open("depression_svm.pkl", "rb") as f:
    depression_model = pickle.load(f)
with open("anxiety_svm.pkl", "rb") as f:
    anxiety_model = pickle.load(f)
with open("stress_svm.pkl", "rb") as f:
    stress_model = pickle.load(f)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", questions=questions)

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve responses from the form as integers
    responses = {}
    for qid in questions.keys():
        responses[qid] = int(request.form.get(qid))  #{"Q1":4,"Q2":2,....,"Q42"}
    
    # Build feature vectors for each subscale based on the defined order
    depression_features = [responses[q] for q in depression_questions]
    anxiety_features    = [responses[q] for q in anxiety_questions]
    stress_features     = [responses[q] for q in stress_questions]
    
    # Convert to numpy arrays and reshape for model prediction
    dep_array = np.array(depression_features).reshape(1, -1)
    anx_array = np.array(anxiety_features).reshape(1, -1)
    str_array = np.array(stress_features).reshape(1, -1)
    
    # Get predictions from the models
    dep_pred = depression_model.predict(dep_array)[0]
    anx_pred = anxiety_model.predict(anx_array)[0]
    str_pred = stress_model.predict(str_array)[0]
    
    # Map numerical predictions to severity labels
    depression_result = severity_mapping.get(dep_pred, "Unknown")
    anxiety_result    = severity_mapping.get(anx_pred, "Unknown")
    stress_result     = severity_mapping.get(str_pred, "Unknown")
    
    return render_template("results.html",
                                  depression_result=depression_result,
                                  anxiety_result=anxiety_result,
                                  stress_result=stress_result)

if __name__ == "__main__":
    app.run(debug=True)