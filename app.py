import os
import re
import requests
import time
import concurrent.futures
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from vehicle_validation import has_vehicle_info, get_missing_info_message

load_dotenv()

static_folder = os.path.abspath('static')
app = Flask(__name__, static_folder=static_folder, static_url_path='/static')
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")
client = OpenAI(api_key=api_key)

# Validate required API keys
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")
if not serpapi_key:
    raise ValueError("SERPAPI_KEY is not set in environment variables")

# 🔧 Smart query cleaner for better search match
def clean_query(text):
    """Enhanced query cleaner for better search match with common automotive parts"""
    original_text = text
    
    # Normalize dashes and remove filler words
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"\bfor\b", "", text, flags=re.IGNORECASE)
    
    # Extract the year, make, and model for later reconstruction
    year_pattern = r'\b(19|20)\d{2}\b'
    year_match = re.search(year_pattern, text)
    year = year_match.group(0) if year_match else ""
    
    # Convert to lowercase for processing
    text_lower = text.lower()
    
    # Expanded part terms dictionary based on your real product data
    part_terms = {
        # Engine components
        "engine wire harness": "engine wiring harness oem",
        "engine wiring harness": "engine wiring harness oem",
        "wire harness": "wiring harness oem",
        "wiring harness": "wiring harness oem",
        "engine mount": "engine motor mount",
        "timing chain kit": "timing chain set complete",
        "timing belt kit": "timing belt kit with water pump",
        "water pump": "water pump with gasket",
        "fuel pump": "fuel pump assembly oem",
        "high pressure fuel pump": "high pressure fuel pump oem",
        "fuel injector": "fuel injectors set",
        "oil pan": "oil pan with gasket set",
        "thermostat": "thermostat with housing",
        "engine splash shield": "engine splash shield underbody cover",
        
        # Transmission components
        "transmission kit": "transmission rebuild kit complete",
        "transmission oil pipe": "transmission oil cooler lines",
        "transmission hose": "transmission cooler line hose",
        "clutch kit": "clutch kit with flywheel",
        
        # Exterior body parts
        "front bumper": "front bumper cover complete assembly",
        "rear bumper": "rear bumper cover complete assembly",
        "bumper assembly": "bumper complete assembly with brackets",
        "chrome bumper": "chrome front bumper complete",
        "fender liner": "fender liner splash shield",
        "headlight": "headlight assembly complete",
        "headlight assembly": "headlight assembly complete oem style",
        "taillight": "tail light assembly complete",
        "grille": "front grille complete assembly",
        "hood": "hood panel assembly",
        "door panel": "door panel complete",
        "mirror": "side mirror assembly",
        "side mirror": "side mirror power assembly complete",
        
        # Brake system
        "brake caliper": "brake caliper with bracket",
        "brake rotor": "brake rotor disc",
        "brake pad": "brake pads set",
        "master cylinder": "brake master cylinder",
        
        # Suspension & Steering
        "control arm": "control arm with ball joint",
        "trailing arm": "trailing arm suspension kit",
        "shock": "shock absorber strut assembly",
        "strut assembly": "strut assembly complete",
        "wheel bearing": "wheel bearing and hub assembly",
        "hub assembly": "wheel hub bearing assembly",
        "power steering": "power steering pump",
        "steering column": "steering column assembly",
        
        # Electrical components
        "alternator": "alternator oem replacement",
        "starter": "starter motor oem",
        "window switch": "power window master switch",
        "master window switch": "power window master control switch",
        
        # HVAC
        "radiator": "radiator complete with fans",
        "radiator assembly": "radiator with fan shroud assembly",
        "ac condenser": "ac condenser with receiver drier",
        "blower motor": "hvac blower motor with fan",
        
        # Wheels
        "rim": "wheel rim replacement",
        "rims": "wheel rims set",
        "hub cap": "hub caps set",
    }
    
    # Check for exact part mentions first
    for part, replacement in part_terms.items():
        # Use word boundaries to match complete terms
        pattern = r'\b' + re.escape(part) + r'\b'
        if re.search(pattern, text_lower):
            # Preserve original capitalization if possible
            text = re.sub(pattern, replacement, text_lower, flags=re.IGNORECASE)
            
            # Make sure year and make/model stay with the part
            if year:
                # Remove year and add it to the beginning
                text = re.sub(year_pattern, "", text)
                text = year + " " + text
                
            # Clean up extra spaces
            text = re.sub(r"\s+", " ", text).strip()
            return text
    
    # If we didn't find a specific part pattern match, try to improve the general query
    if "oem" not in text_lower and "aftermarket" not in text_lower:
        # Add OEM for better quality results if it's not an aftermarket search
        if any(word in text_lower for word in ["genuine", "original", "factory"]):
            text += " OEM genuine"
        elif "assembly" in text_lower or "complete" in text_lower:
            text += " complete assembly"
    
    # Trim and clean spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# VIN decoder helper function with caching (VINs don't change, so cache indefinitely)
@lru_cache(maxsize=500)
def decode_vin(vin):
    """Decode VIN with caching for better performance"""
    if not vin:
        return {}
    try:
        url = f'https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvaluesextended/{vin}?format=json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        if data and data.get('Results') and len(data['Results']) > 0:
            result = data['Results'][0]
            if result.get('Make') and result.get('ModelYear'):
                return result
            else:
                print(f"Invalid VIN data received: Missing Make or ModelYear for VIN {vin}")
        else:
            print(f"No results found for VIN {vin}")
    except requests.exceptions.RequestException as e:
        print(f"VIN decode request error: {e}")
    except ValueError as e:
        print(f"VIN decode JSON parsing error: {e}")
    except Exception as e:
        print(f"VIN decode unexpected error: {e}")
    return {}

# Cache results for 5 minutes (300 seconds)
@lru_cache(maxsize=100)
def get_serpapi_cached(query_type, query, timestamp):
    """Cache wrapper for SerpAPI requests. Timestamp is used to invalidate cache after 5 minutes."""
    params = {
        "engine": "ebay",
        "ebay_domain": "ebay.com",
        "_nkw": query,
        "LH_ItemCondition": "1000" if query_type == "new" else "3000",
        "LH_BIN": "1",  # Buy It Now only
        "api_key": serpapi_key
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {query_type} items from SerpAPI: {e}")
        return {"organic_results": []}

def fetch_ebay_results(query_type, query, timestamp):
    """Function to fetch eBay results for concurrent execution"""
    results = get_serpapi_cached(query_type, query, timestamp)
    return process_ebay_results(results, query, max_items=3)

def get_ebay_serpapi_results(query):
    """Fetch eBay results using concurrent requests"""
    # Use timestamp for cache invalidation every 5 minutes
    cache_timestamp = int(time.time() / 300)
    
    # Define tasks for concurrent execution
    tasks = [
        ("new", query, cache_timestamp),
        ("used", query, cache_timestamp)
    ]
    
    all_items = []
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Create a mapping of futures to their tasks
        future_to_task = {
            executor.submit(fetch_ebay_results, *task): task[0]
            for task in tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task_type = future_to_task[future]
            try:
                items = future.result()
                all_items.extend(items)
            except Exception as e:
                print(f"Error processing {task_type} items: {e}")
    
    return all_items

def process_ebay_results(results, query, max_items=3):
    """Helper function to process eBay results"""
    processed_items = []
    keywords = query.lower().split()
    
    for item in results.get("organic_results", []):
        if len(processed_items) >= max_items:
            break
            
        title = item.get("title", "").lower()
        if any(kw in title for kw in keywords):
            # Extract price
            price = "Price not available"
            if isinstance(item.get("price"), dict):
                price = item.get("price", {}).get("raw", "Price not available")
            else:
                price = item.get("price", "Price not available")
                
            # Extract shipping
            shipping = "Shipping not specified"
            if isinstance(item.get("shipping"), dict):
                shipping = item.get("shipping", {}).get("raw", "Shipping not specified")
            else:
                shipping = item.get("shipping", "Shipping not specified")
                
            # Extract condition
            condition = item.get("condition", "Not specified")
                
            processed_items.append({
                "title": item.get("title"),
                "price": price,
                "shipping": shipping,
                "condition": condition,
                "link": item.get("link")
            })
    
    return processed_items
# Main GPT Assistant route - original version for regular form submission
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Regular form submission just renders the template
        return render_template("index.html")
    return render_template("index.html")

# Sanitize user input
def sanitize_input(text):
    if not text:
        return ""
    # Remove any potentially harmful characters
    sanitized = re.sub(r'[^\w\s\-.,?!@#$%^&*()_+=[\]{}|:;<>"/]', '', text)
    return sanitized.strip()

# AJAX endpoint for GPT Assistant
@app.route("/api/search", methods=["POST"])
def search_api():
    query = sanitize_input(request.form.get("prompt", ""))
    
    if not query:
        return jsonify({
            "success": False,
            "validation_error": "Please enter a valid search query."
        })
    
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

        if search_lines:
            # Extract the search term without the emoji
            search_term_raw = search_lines[0].replace("🔎", "").strip()
            
            # Clean and optimize the search term
            search_term = clean_query(search_term_raw)
            
            # If there's a second search term, use it as a fallback
            fallback_term = None
            if len(search_lines) > 1:
                fallback_term = clean_query(search_lines[1].replace("🔎", "").strip())
        else:
            # Fallback to user's query if no GPT search term
            search_term = clean_query(query)
            fallback_term = None

        # Fetch primary search results
        listings = get_ebay_serpapi_results(search_term)

        # If we didn't get good results with the first term and have a fallback, try it
        if len(listings) < 2 and fallback_term:
            fallback_listings = get_ebay_serpapi_results(fallback_term)
            # Add any new listings not already present
            existing_titles = [item.get('title', '').lower() for item in listings]
            for item in fallback_listings:
                if item.get('title', '').lower() not in existing_titles:
                    listings.append(item)

        return jsonify({
            "success": True,
            "questions": questions,
            "listings": listings
        })
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            "success": False,
            "error": "An error occurred while processing your request. Please try again later."
        })
# AJAX endpoint for VIN decoding
@app.route("/api/vin-decode", methods=["POST"])
def vin_decode_api():
    vin = sanitize_input(request.form.get("vin", ""))
    
    if not vin:
        return jsonify({"error": "No VIN provided"})
    
    # Validate VIN format (17 alphanumeric characters for modern vehicles)
    if not re.match(r'^[A-HJ-NPR-Z0-9]{17}$', vin):
        return jsonify({"error": "Invalid VIN format. VIN should be 17 alphanumeric characters (excluding I, O, and Q)."})
    
    try:
        vin_info = decode_vin(vin)
        
        if not vin_info or not vin_info.get('Make'):
            return jsonify({"error": "Could not decode VIN. Please check the VIN and try again."})
        
        # Return the VIN information as JSON
        return jsonify(vin_info)
    except Exception as e:
        # Log the error but don't expose details to the client
        print(f"VIN decode error: {e}")
        return jsonify({"error": "An error occurred while decoding the VIN. Please try again later."})

# For backward compatibility - redirect old routes to the API endpoints
@app.route("/vin-decode", methods=["POST"])
def vin_decode():
    return vin_decode_api()

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5040))
    app.run(host="0.0.0.0", port=port)