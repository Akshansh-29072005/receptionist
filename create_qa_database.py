#!/usr/bin/env python3
"""
Script to create a sample Q&A database in Excel format.
Run this to generate qa_database.xlsx with sample college Q&A data.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "qa_database.xlsx")
 
# Sample data with columns: question_en, answer_en, question_hi, answer_hi, keywords
sample_data = [
    {
        "question_en": "What is the capital of India?",
        "answer_en": "The capital of India is New Delhi.",
        "question_hi": "भारत की राजधानी क्या है?",
        "answer_hi": "भारत की राजधानी नई दिल्ली है।",
        "keywords": "capital, India, New Delhi"
    },
    {
        "question_en": "What is the boiling point of water?",
        "answer_en": "The boiling point of water is 100 degrees Celsius.",
        "question_hi": "पानी का उबलने का तापमान क्या है?",
        "answer_hi": "पानी का उबलने का तापमान 100 डिग्री सेल्सियस है।",
        "keywords": "boiling point, water, temperature"
    },
    {
        "question_en": "What is your name",
        "answer_en": "I am Shankra Mitra An A-I receptionist robot of S-S-I-P-M-T, Raipur.",
        "question_hi": "तुम्हारा नाम क्या है?",
        "answer_hi": "मेरा नाम शंकरा मित्रा है, मैं S-S-I-P-M-T, रायपुर का एक ए-आई रिसेप्शनिस्ट रोबोट हूँ।",
        "keywords": "name, introduction, robot"
    },
    {
        "question_en": "Who is the principal of the college",
        "answer_en": "The principal of the college is Dr. Alok Kumar Jain.",
        "question_hi": "कॉलेज के प्रिंसिपल कौन हैं?",
        "answer_hi": "कॉलेज के प्रिंसिपल डॉक्टर आलोक कुमार जैन हैं।",
        "keywords": "principal, college, Dr. Alok Kumar Jain"
    },
    {
        "question_en": "Who is the chairman of the college",
        "answer_en": "The chairman of the college is Mr. Nishant Tripathi.",
        "question_hi": "कॉलेज के चेयरमैन कौन हैं?",
        "answer_hi": "कॉलेज के चेयरमैन श्री निशांत त्रिपाठी हैं।",
        "keywords": "chairman, college, Mr. Nishant Tripathi"
    },
    {
        "question_en": "What courses does the college offer?",
        "answer_en": "The college offers undergraduate and postgraduate courses in engineering, management, and computer applications.",
        "question_hi": "कॉलेज कौन-कौन से कोर्स ऑफर करता है?",
        "answer_hi": "कॉलेज इंजीनियरिंग, प्रबंधन, और कंप्यूटर अनुप्रयोगों में स्नातक और स्नातकोत्तर कोर्स ऑफर करता है।",
        "keywords": "courses, college, engineering, management, computer applications"
    },
    {
        "question_en": "What are the college's admission criteria?",
        "answer_en": "The college's admission criteria include academic performance, entrance exam scores, and extracurricular activities.",
        "question_hi": "कॉलेज के प्रवेश मानदंड क्या हैं?",
        "answer_hi": "कॉलेज के प्रवेश मानदंडों में शैक्षणिक प्रदर्शन, प्रवेश परीक्षा के अंक, और सह-पाठ्यक्रम गतिविधियाँ शामिल हैं।",
        "keywords": "admission criteria, college, academic performance, entrance exam"
    },
    {
        "question_en": "What facilities are available on campus?",
        "answer_en": "The campus offers facilities such as a library, sports complex, canteen, and hostels.",
        "question_hi": "कैंपस में कौन-कौन सी सुविधाएँ उपलब्ध हैं?",
        "answer_hi": "कैंपस में पुस्तकालय, खेल परिसर, कैफेटेरिया, और हॉस्टल जैसी सुविधाएँ उपलब्ध हैं।",
        "keywords": "facilities, campus, library, sports complex, cafeteria, hostels"
    },
    {
        "question_en": "What extracurricular activities does the college offer?",
        "answer_en": "The college offers various extracurricular activities including sports, cultural events, and technical workshops.",
        "question_hi": "कॉलेज कौन-कौन सी सह-पाठ्यक्रम गतिविधियाँ ऑफर करता है?",
        "answer_hi": "कॉलेज विभिन्न सह-पाठ्यक्रम गतिविधियाँ ऑफर करता है जिनमें खेल, सांस्कृतिक कार्यक्रम, और तकनीकी कार्यशालाएँ शामिल हैं।",
        "keywords": "extracurricular activities, college, sports, cultural events, technical clubs"
    },
    {
        "question_en": "How can I contact the college administration?",
        "answer_en": "You can contact the college administration via phone, email, or by visiting the administrative office on campus.",
        "question_hi": "मैं कॉलेज प्रशासन से कैसे संपर्क कर सकता हूँ?",
        "answer_hi": "आप फोन, ईमेल के माध्यम से या कैंपस में प्रशासनिक कार्यालय का दौरा करके कॉलेज प्रशासन से संपर्क कर सकते हैं।",
        "keywords": "contact, college administration, phone, email, office"
    },
    {
        "question_en": "Who is the addmission incharge of the college",
        "answer_en": "The admission incharge of the college is Mr. Atul Chakrawarti.",
        "question_hi": "कॉलेज के एडमिशन इंचार्ज कौन हैं?",
        "answer_hi": "कॉलेज के एडमिशन इंचार्ज श्री अतुल चक्रवर्ती हैं।",
        "keywords": "admission incharge, college, Mr. Atul Chakrawarti"
    },
    {
        "question_en": "What is the location of the college?",
        "answer_en": "The college is located in Raipur, Chhattisgarh, India.",
        "question_hi": "कॉलेज का स्थान क्या है?",
        "answer_hi": "कॉलेज रायपुर, छत्तीसगढ़, भारत में स्थित है।",
        "keywords": "location, college, Raipur, Chhattisgarh"
    },
]
# Create DataFrame and save to Excel
df = pd.DataFrame(sample_data)
df.to_excel(EXCEL_PATH, index=False)

print(f"✓ Q&A database created: {EXCEL_PATH}")
print(f"✓ Total entries: {len(df)}")
print(f"\nColumns: question_en, answer_en, question_hi, answer_hi, keywords")