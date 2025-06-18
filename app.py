from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
from flask_cors import CORS
from deep_translator import GoogleTranslator
from cachetools import TTLCache
import pycountry
import asyncio
from deep_translator import GoogleTranslator
import langcodes

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set your Google AI API key
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY"))

# Set up cache
cache = TTLCache(maxsize=100, ttl=300)  # LRU Cache with TTL
CACHE_TIMEOUT = timedelta(minutes=5)

# Translation cache
translation_cache = TTLCache(maxsize=1000, ttl=CACHE_TIMEOUT.total_seconds())
LANGUAGE_ALIASES = {
    "Punjabi": "pa",
    # Add more aliases if needed
}

# Compile regex patterns once
mcq_pattern = re.compile(
    r'\{\s*"question":\s*"([^"]*)",\s*'
    r'"option1":\s*"([^"]*)",\s*'
    r'"option2":\s*"([^"]*)",\s*'
    r'"option3":\s*"([^"]*)",\s*'
    r'"option4":\s*"([^"]*)",\s*'
    r'"answer":\s*"([^"]*)"\s*\}',
    re.DOTALL,
)
essay_pattern = re.compile(
    r'"question":\s*"([^"]+)"',
    re.DOTALL,
)
tf_pattern = re.compile(
    r'\{\s*"question":\s*"([^"]*)",\s*'
    r'"answer":\s*"([^"]*)"\s*\}',
    re.DOTALL,
)


LANGUAGE_ALIASES = {
    "punjabi": "pa",
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "hindi": "hi",
    "arabic": "ar",
    "sindhi": "sd",  # Added Sindhi
    # Add more languages as needed
}

def get_language_code(language_name):
    # Check if the language name exists in LANGUAGE_ALIASES
    if language_name in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[language_name]

    # Use langcodes to get the ISO code if it's not in aliases
    try:
        lang = langcodes.get(language_name)
        return lang.language
    except Exception:
        return None

def chunk_text(text, chunk_size=5000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def translate_sentence(sentence, target_language):
    # Get the target language code
    target_lang_code = get_language_code(target_language.lower())

    if not target_lang_code:
        return f"Language '{target_language}' is not supported."

    try:
        # If sentence is too long, split it into smaller chunks
        if len(sentence) > 5000:
            chunks = chunk_text(sentence)
            translated_chunks = [
                GoogleTranslator(source='auto', target=target_lang_code).translate(chunk)
                for chunk in chunks
            ]
            return ''.join(translated_chunks)
        else:
            return GoogleTranslator(source='auto', target=target_lang_code).translate(sentence)
    except Exception as e:
        return f"An error occurred: {e}"


def generate_question_and_answer(class_name, course_name, section, subsection, language, question_type, Difficulty):
    """Generates questions and answers using the Gemini Pro model."""
    if question_type == "true false":
        pattern = tf_pattern
        prompt = f"""Design a true/false quiz for {course_name} {class_name} studying {subsection}. The quiz should focus on {section}. Questions should be at a {Difficulty} level and written in {language}.

The output should be in JSON format with a key named "questions", containing an array of objects where each object has a "question" key for the question text and an "answer" key for the correct answer (either "true" or "false").

Here is an example of the expected output format:
{{
  "questions": [
    {{
      "question": "The derivative of ln(x + 1) is 1/(x + 1).",
      "answer": "true"
    }},
    {{
      "question": "The derivative of e^x is xe^x.",
      "answer": "false"
    }},
    {{
      "question": "The derivative of sin(x) is cos(x).",
      "answer": "true"
    }}
  ]
}}

Provide a total of 25 questions."""

    elif question_type == "essay":
        prompt = f"""Design 25 essay-type questions for {course_name} {class_name} studying {subsection}. Each question should focus on {section} at a {Difficulty} level to understand, written in {language}. Format the output as JSON, with each question structured under the heading "question" as an array of objects, where each object has a single "question" key.

Example format:
{{
  "question": [
    {{
      "question": "Prove the identity: (sin A + cos A)^2 + (sin A - cos A)^2 = 2."
    }},
    {{
      "question": "Simplify the expression: (tan A - cot A)/(tan A + cot A) + (tan A + cot A)/(tan A - cot A)."
    }},
    {{
      "question": "Solve the equation: 2 cos^2 A - 3 sin A cos A + 1 = 0."
    }},
    ...
  ]
}}
"""


        pattern = essay_pattern
    else:
        pattern = mcq_pattern
        prompt = f"""Design 25 multiple-choice type questions for {course_name} {class_name} studying {subsection}. Each question should focus on {section} at a {Difficulty} level to understand, written in {language}. Format the output as JSON, with each question containing the keys "question", "option1", "option2", "option3", "option4", and "answer" (where "answer" is the correct answer text, not an option label).

Example format:
{{
  "questions": [
    {{
      "question": "Integrate the function (x^2 + 2x - 3) dx with respect to x.",
      "option1": "x^3/3 + x^2 - 3x + C",
      "option2": "x^3/3 + 2x^2 - 3x + C",
      "option3": "2x^3/3 + x^2 - 3x + C",
      "option4": "x^3/3 + 4x^2 - 3x + C",
      "answer": "x^3/3 + x^2 - 3x + C"
    }},
    {{
      "question": "Evaluate the integral ∫(e^x + sin(x)) dx.",
      "option1": "e^x - cos(x) + C",
      "option2": "e^x + cos(x) + C",
      "option3": "2e^x - sin(x) + C",
      "option4": "2e^x + cos(x) + C",
      "answer": "e^x - cos(x) + C"
    }},
    ...
  ]
}}
"""

    model = genai.GenerativeModel(model_name="gemini-pro")
    response = model.generate_content(prompt)
    generated_text = response.text
    print(generated_text)

    matches = pattern.findall(generated_text)
    print(matches)
    mcq_data = []

    for match in matches:
        if question_type == "true false":
            mcq_data.append({
                "description": match[0],
                "answer": match[1],
            })
        elif question_type == "essay":
            mcq_data.append({
                "description": match,
            })
        else:
            mcq_data.append({
                "description": match[0],
                "options": match[1:5],
                "answer": match[5],
            })

    return mcq_data


async def process_questions(data):
    """Process questions asynchronously."""
    global cache  # Declare as global to modify the global variable
    global translation_cache  # Declare as global to modify the global variable

    class_name = data.get("className", "")
    course_name = data.get("courseName", "")
    section = data.get("sectionName", "")
    subsection = data.get("subSectionName", "")
    language = data.get("languageName", "")
    language1 = data.get("languageName1", None)
    question_type = data.get("type", "")
    Difficulty = data.get("difficultyName", "")

    if question_type=="short":
        question_type = "essay"

    if not all([class_name, course_name, section, subsection, language, question_type, Difficulty]):
        return {"error": "Missing data"}, 400

    cache_key = (class_name, course_name, section, subsection, language, question_type, Difficulty)

    # Clean expired cache entries
    now = datetime.now()
    cache = {key: value for key, value in cache.items() if now - value[1] <= CACHE_TIMEOUT}

    # Get the previous questions
    previous_questions, _ = cache.get(cache_key, (set(), datetime.now()))
    unique_questions = set()
    unique_answers = []
    unique_option = []
    for _ in range(20):  # Limit the number of retries
        mcq_data = generate_question_and_answer(class_name, course_name, section, subsection, language, question_type, Difficulty)
        print(mcq_data)
        should_break_outer = False

        for item in mcq_data:
            q = item["description"]
            if question_type =="true false" or question_type == "mcq":
                a = item["answer"]
                if question_type != "true false":
                    o = item["options"]

            if q not in previous_questions and len(unique_questions) < 10:
                unique_questions.add(q)
                if question_type =="true false" or question_type == "mcq":
                    unique_answers.append(a)
                    if question_type != "true false":
                        unique_option.append(o)
                previous_questions.add(q)
            if len(unique_questions)>=10:
                should_break_outer = True
                break
        if should_break_outer:
            break
    if unique_questions:
        cache[cache_key] = (previous_questions, datetime.now())
        lang1 = []
        if question_type.lower() == "mcq":
            for q, a, o in zip(unique_questions, unique_answers, unique_option):
                if language1 is not None:
                    ques = translate_sentence(q, language1)
                    ans = translate_sentence(a, language1)
                    opt = [translate_sentence(option, language1) for option in o]
                    lang1.append({
                    "description": q,
                    "description1": ques,
                    "options": o,
                    "options1": opt,
                    "answer": a,
                    "answer1": ans,

                })
                else:
                    lang1.append({
                    "description": q,
                    "options": o,
                    "answer": a,

                })

        elif question_type.lower() in ["short", "true false"]:
            for q, a in zip(unique_questions, unique_answers):
                if language1 is not None:
                    ques = translate_sentence(q, language1)
                    ans = translate_sentence(a, language1)
                    lang1.append({"answer": a,
                                "answer1": ans,
                                "description": q,
                                "description1": ques
                                })
                else:
                    lang1.append({"answer": a,
                                "description": q,
                                })


        elif question_type.lower() == "essay":
            for q in unique_questions:
                if language1 is not None:
                    ques = translate_sentence(q, language1)
                    lang1.append({"description": q,
                                "description1": ques})
                else:
                    lang1.append({"description": q})


        return {"result": lang1, "message": "all questions", "success": True}, 200


      # Short delay to avoid rapid retries
    return {"success": False, "message": "No new unique questions found after multiple attempts"}, 500

@app.route("/generateQuestionsUsingAi", methods=["POST"])
def generate_question_endpoint():
    """API endpoint to generate questions and answers."""
    data = request.json
    response, status_code = asyncio.run(process_questions(data))
    return jsonify(response), status_code

@app.route("/hello",methods=["GET"])
def hello():
    return jsonify({"message":"hello"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=True)