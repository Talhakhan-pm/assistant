import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from vehicle_validation import has_vehicle_info, get_missing_info_message

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")
client = OpenAI(api_key=api_key)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "auto-parts-assistant-key")
print("🧪 SERPAPI_KEY loaded:", serpapi_key)

# 🔧 Smart query cleaner for better search match
def clean_query(text):
    # Normalize dashes
    text = re.sub(r"[–—]", "-", text)

    # Remove weak filler words
    text = re.sub(r"\bfor\b", "", text, flags=re.IGNORECASE)

    # Replace vague phrases with stronger keywords
    text = text.lower()
    text = text.replace("oem timing kit", "timing belt kit with water pump")
    text = text.replace("timing kit", "timing belt kit with water pump")
    text = text.replace("assembly", "complete assembly")
    text = text.replace("radiator assembly", "radiator with fan and shroud")

    # Trim and clean spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# VIN decoder helper function
def decode_vin(vin):
    if not vin:
        return {}
    try:
        url = f'https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvaluesextended/{vin}?format=json'
        response = requests.get(url)
        data = response.json()
        if data and data['Results']:
            result = data['Results'][0]
            if result.get('Make') and result.get('ModelYear'):
                return result
    except Exception as e:
        print(f"VIN decode error: {e}")
    return {}

# eBay SerpAPI listing fetcher
def get_ebay_serpapi_results(query):
    print("🔥 SerpAPI called with:", query)

    url = "https://serpapi.com/search"
    params = {
        "engine": "ebay",          # Change from "google_shopping" to "ebay"
        "ebay_domain": "ebay.com", # Specify eBay domain
        "_nkw": query,
        "api_key": serpapi_key
    }
    print("📦 Final request params:", params)
    try:
        response = requests.get(url, params=params)
        results = response.json()
        print("🔍 SerpAPI response:", results)

        top_results = []
        keywords = query.lower().split()
        # Adjust the parsing based on eBay response structure
        for item in results.get("organic_results", []):
            title = item.get("title", "").lower()
            if any(kw in title for kw in keywords):
                top_results.append({
                    "title": item.get("title"),
                    "price": item.get("price", {}).get("raw"),
                    "link": item.get("link")
                })
            if len(top_results) >= 5:
                break

        return top_results
    except Exception as e:
        print("SerpAPI error:", e)
        return []

# Main GPT Assistant route - original version for regular form submission
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Regular form submission just renders the template
        return render_template("index.html")
    return render_template("index.html")

# AJAX endpoint for GPT Assistant
@app.route("/api/search", methods=["POST"])
def search_api():
    query = request.form.get("prompt", "").strip()
    
    # Check if query has sufficient vehicle information
    if not has_vehicle_info(query):
        validation_error = get_missing_info_message(query)
        return jsonify({
            "success": False,
            "validation_error": validation_error
        })

    prompt = f"""
You are an auto parts fitment expert working for a US-based parts sourcing company. The goal is to help human agents quickly identify the correct OEM part for a customer's vehicle.

The customer just said: "{query}"

Do not provide explanations, summaries, or filler text. Format everything in direct, clean bullet points.

Your job is to:
1. If anything is misspelled, auto-correct it to the best of your automotive knowledge.

2. Validate:
   - Confirm the vehicle exists in the US market.
   - If invalid, return:
     👉 This vehicle is not recognized in US-spec. Please clarify. Did you mean one of these: [list 2–3 real US models for that year/make]?
   - If valid, do NOT confirm it in a sentence. Just move on.

3. If valid:
   - List factory trims with engine options (displacement + type + drivetrain).
   - DO NOT repeat parsed info in sentence form

4. Ask follow-up questions, max 3:
   - Question 1: Ask about directly associated hardware needed (e.g., bumper → brackets, fog trims, sensors if applicable)
   - Question 2: Only ask follow-up if something affects fitment — like transmission type, submodel, or drivetrain. 
    Do NOT ask vague or unnecessary questions like modifications or preferences.
   - Fitment: If fitment is shared across multiple years, mention the range with platform/chassis code — you can take a guess if needed. Just say it out. No worries. 
   - If more products are involved, you can ask more questions, max 2.

5. Finish with a bolded search-optimized lookup phrase, (add a emoji of world right before the phrase):
   - Format: lowercase string including [year or range] + make + model + trim (if needed) + engine (if relevant) + oem + part name
   - Think of it as a search term for a customer to find the part. Use the most relevant keywords. Give two search terms for the same part with another name.
   - Example 1:  "🔎 2020–2022 honda civic ex oem front bumper"
   - Example 2: "🔎 2020 – 2022 honda civic ex oem bumper cover"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        questions = response.choices[0].message.content.strip()

        # Use GPT-generated search term if available, else fallback to cleaned query
        search_lines = [line for line in questions.split("\n") if "🔎" in line]
        search_term = clean_query(search_lines[0].replace("🔎", "").strip()) if search_lines else clean_query(query)
        print("🧼 Final search term used:", search_term)

        listings = get_ebay_serpapi_results(search_term)

        return jsonify({
            "success": True,
            "questions": questions,
            "listings": listings
        })
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

# AJAX endpoint for VIN decoding
@app.route("/api/vin-decode", methods=["POST"])
def vin_decode_api():
    vin = request.form.get("vin", "").strip()
    
    if not vin:
        return jsonify({"error": "No VIN provided"})
    
    try:
        vin_info = decode_vin(vin)
        
        if not vin_info or not vin_info.get('Make'):
            return jsonify({"error": "Could not decode VIN. Please check the VIN and try again."})
        
        # Return the VIN information as JSON
        return jsonify(vin_info)
    except Exception as e:
        print(f"VIN decode error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"})

# For backward compatibility - redirect old routes to the API endpoints
@app.route("/vin-decode", methods=["POST"])
def vin_decode():
    return vin_decode_api()

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5040))
    app.run(host="0.0.0.0", port=port)