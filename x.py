import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import uuid
import asyncio
import torch
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import edge_tts

# ---------------- ENV SETUP ----------------
AutoConfig.register("offline", AutoConfig)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---------------- MODEL LOAD ----------------
model_path = "./flan_t5_base_local"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32
).to(device)

model.eval()
print(f"Model loaded on: {device}")

# ---------------- KNOWLEDGE ----------------
paragraph = """
Welcome! I’m here to assist you with accurate information, clear guidance, and friendly support. Feel free to ask anything—whether it’s a simple question or a complex topic, I’ll do my best 
to help you. Let’s begin your experience.
Hello! It’s great to have you here. I’m your virtual assistant, always ready to help you 
learn, explore, and solve problems. How can I assist you today?
Thank you for interacting with me. I hope I was able to support you effectively. 
If you ever need guidance, clarification, or information, just return anytime—I'm always here 
to help!
Thank you for your time! Wishing you a wonderful day ahead. If you need assistance again, 
don’t hesitate to return. Goodbye!
I’m sorry, I couldn’t understand that clearly. Could you please rephrase your question? 
I want to make sure I give you the correct information.
Thank you for your question! I appreciate your curiosity and engagement. If you have more 
questions or need further explanation, feel free to ask anytime.
Shri Shankaracharya Institute of Professional Management and Technology,
SSIPMT, Raipur is Located in Raipur, established on 8 August 2008.
SSIPMT offers B Tech, M Tech, MBA and Ph D. courses
SSIPMT is affiliated to CSVTU, Bhilai and approved by AICTE, New Delhi.
SSIPMT NAAC, A+ Grade, Four-star rating in IIC college.
SSIPMT has India’s first AICTE IDEA Lab.
SSIPMT located, 10 Kilometer from Swami Vivekananda Airport,
Raipur, 13 Kilometer away from the center of the city, Ghadi Chowk and 16
Kilometer from Raipur railway station.
The Chairman of board of Governors, is Shri Nishant Tripathi.
Principal of SSIPMT is Doctor Alok Kumar Jain.
SSIPMT has central library with dedicated sections for
reference, text books, reading, newspapers and circulation.
SSIPMT central library has 5625 E-Journals, 28407 E Books 
SSIPMT offers B.Tech. programs in Computer Science and Engineering,Information Technology,
Artificial Intelligence and Machine Learning,Data Science,Civil Engineering, Mechanical Engineering, and Electronics and Telecommunication Engineering,
SSIPMT offers Master of Technology, M.Tech programs in Structural
Engineering, Computer Science Engineering with specialization in Artificial Intelligence and Machine Learning
SSIPMT offers Master in Business Administration, MBA program with specializations in Marketing, Finance, Human Resource Management, Production and Operations Management, and Systems Management.
SSIPMT offers PhD programs in Management and Engineering and Applied
Sciences for scholars interested in advanced research, innovation, and academic
careers.
   Fees for B.Tech. program is Rupees 42950 per semester.
Fees for M.Tech. program is Rupees 39250 per semester.
Fees for MBA program is Rupees 37300 per semester. 
Fees for PhD is Rupees 35750 per semester.
SSIPMT has strong placements.
At SSIPMT the highest package has been rupees 42 Lakhs Per Annum and average package of rupees 7 point 5 Lakhs Per
Annum.
SSIPMT hosts companies like Adobe, Microsoft, Hacker Rank, SAP, Tech. Mahindra, TCS, Infosys, IBM, L and T, and UltraTech 
Dr. Yogesh Rathore is Dean, Training and Placements and his contact number is 9691990198.
For more details about the training and placements contact Dr. Yogesh Rathore at 9691990198.
SSIPMT has separate hostels for boys and girls.
SSIPMT hostels have free Wi Fi, power backup, sports facilities, and vegetarian meals served four times a day.
SSIPMT hostel fees is rupees 66400 per year.
SSIPMT operates 21 buses covering all major areas of Raipur city and Bhilai.
SSIPMT Bus fees per year is rupees 20000 from Raipur to SSIPMT Raipur and rupees 22000 from Bhilai to SSIPMT Raipur.

SSIPMT Principal is Dr. Alok Kumar Jain.
Head or H.O.D. of Department of Computer Science and Engineering is Dr. Anand Tamrakar.
Head or H.O.D. of Department of Artificial Intelligence and Machine Learning is Dr. Rudra Pratap Singh Chauhan.
Head or H.O.D. of Department of Computer Science and Engineering is
Head or H.O.D. of Department of Data Science is Dr. Abhishek Badholia.
Head or H.O.D. of Department of Information Technology is Sunil Dewangan.
Head or H.O.D. of Department of Electronics and Telecommunication is Dr. Hemlata Sinha.
Head or H.O.D. of Department of Civil Engineering is Dr. Tarun Kumar Rajak.
Head or H.O.D. of Department of Mechanical Engineering is Mr. Aakash Soni.
Head or H.O.D. of Department of Management Studies or MBA is Dr. Sapna Sharma.
SSIPMT has sports facilities, including badminton and table tennis, indoor games such as chess and carom.
Admissions at SSIPMT are conducted through online counselling by the
Directorate of Technical Education, Government of Chhattisgarh. 
For admission to B.Tech. programs, candidates must have appeared in CGPET or JEE entrance
examination and passed Class 12, with a minimum of 45 percent marks with Physics, Chemistry, and Mathematics.
Admission to the MBA program requires a valid score in Common Admission Test or CAT, Management Aptitude Test or MAT, Common Management Admission Test or CMAT, Xavier Aptitude Test or XAT, or ATMA, along with a three year bachelor’s degree with at least 50 percent marks.
Admission to the M. Tech program is offered through two categories. Under the Sponsored category, candidates must have a minimum of two years of work experience along with a sponsorship letter from their employer. The second
category is based on merit through a valid Graduate Aptitude Test in Engineering, GATE score.
For any details about admission at SSIPMT please contact Atul Chakrawarti, Mr. Navdeep Khare, Mr.
Krishna Kumar Dewangan, Mr. Anant Kumar. 
The contact numbers are Mr. Atul Chakrawarti, 9617020000, Mr. Navdeep Khare,
9522219177, 8878444000, Mr. Krishna Kumar Dewangan, 9691787970, Mr.
Anant Kumar, 7999712023.

SSIPMT Spellbinders is an extracurricular club of the institute that focuses on public speaking, leadership, communication, and confidence building.
Spellbinders is affiliated with Toastmasters International and follows its structured learning system. The club is open only to students and faculty members of SSIPMT Raipur.
SSIPMT Toastmaster club in charge is Dr. Seema Arora.
For any other information please contact Miss Mohini at 9754406568.
"""

# ---------------- INDIAN VOICE ----------------
VOICE = "en-IN-NeerjaNeural"   # Female Indian
#VOICE = "en-IN-PrabhatNeural"  # Male Indian

def humanize_for_tts(text):
    text = text.replace(". ", ".\n")
    text = text.replace("SSIPMT", "S S I P M T")
    text = text.replace("AICTE", "A I C T E")
    text = text.replace("NAAC", "N A A C")
    text = text.replace("PhD", "P H D")
    text = text.replace("MBA", "M B A")
    return text

async def speak_async(text, output_file):
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE
    )
    await communicate.save(output_file)

def speak_text(text, output_file):
    asyncio.run(speak_async(text, output_file))

# ---------------- TEXT GENERATION ----------------
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=180,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, 99

# ---------------- CLEANUP OLD AUDIO ----------------
def cleanup_old_audio(folder="static", keep_last=5):
    files = sorted(
        [f for f in os.listdir(folder) if f.startswith("answer_")],
        key=lambda x: os.path.getmtime(os.path.join(folder, x)),
        reverse=True
    )
    for f in files[keep_last:]:
        os.remove(os.path.join(folder, f))

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = conf = audio_file = None

    if request.method == "POST":
        question = request.form["question"]

        prompt = f"""
        Use only the information below.

        Paragraph:
        \"\"\"{paragraph}\"\"\"

        Question: {question}
        Answer:
        """

        answer, conf = generate_answer(prompt)

        clean_answer = humanize_for_tts(answer)

        # 🔥 UNIQUE AUDIO FILE (FIXES OVERLAP)
        audio_file = f"answer_{uuid.uuid4().hex}.wav"
        audio_path = os.path.join("static", audio_file)

        speak_text(clean_answer, audio_path)
        cleanup_old_audio()

    return render_template(
        "index.html",
        answer=answer,
        conf=conf,
        audio_file=audio_file
    )

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)