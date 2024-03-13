from flask import Flask, request, jsonify, render_template, send_from_directory
import pytesseract
import openai
from PIL import Image
from dotenv import load_dotenv
import base64
import os
import cv2
import numpy as np
import io
import json

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

app = Flask(__name__)
load_dotenv()  # Load environment variables from .env file

# Ensure your .env file contains OPENAI_API_KEY variable
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/recipes/<filename>')
def custom_static(filename):
    return send_from_directory('recipes', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_stream = file.read()
        npimg = np.frombuffer(file_stream, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image and get debug images
        preprocessed_image, debug_images = preprocess_image_for_ocr(image, debug=True)

        extracted_text = pytesseract.image_to_string(preprocessed_image, lang='eng')
        print(extracted_text)

        parsed_recipe = parse_recipe_with_openai(extracted_text)
        
        try:
            recipe_data = json.loads(parsed_recipe)
            recipe_name = recipe_data["name"].replace(" ", "_").replace("/", "_").lower()
            filename = f"{recipe_name}.html"
            filepath = os.path.join('recipes', filename)
            
            if not os.path.exists('recipes'):
                os.makedirs('recipes')
            
            save_recipe_as_html(recipe_data, filepath)
            
            return jsonify({'message': 'Success', 'filepath': f'/recipes/{filename}'})
        except Exception as e:
            return jsonify({'error': f'Failed to process and save the recipe. Error: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Preprocessing failed.'}), 500

def convert_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess_image_for_ocr(image, debug=True):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    
    # Optional: Apply gamma correction to lighten/darken the image
    gamma = 7.0  # Adjust this value based on your needs, > 1 to lighten, < 1 to darken
    gamma_corrected_image = np.array(255 * (clahe_image / 255) ** gamma, dtype='uint8')
    
    # Apply adaptive thresholding to binarize the image
    adaptive_thresh_image = cv2.adaptiveThreshold(gamma_corrected_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    
    if debug:
        # Collect images for debugging
        debug_images = {
            'gray_image': gray_image,
            'clahe_image': clahe_image,
            'gamma_corrected_image': gamma_corrected_image,
            'adaptive_thresh_image': adaptive_thresh_image,
        }
        #return adaptive_thresh_image, debug_images
        return image, debug_images
    else:
        #return adaptive_thresh_image
        return image

def parse_recipe_with_openai(text):
    gpt_assistant_prompt = "Parse the following recipe text into structured information including the title, ingredients, preparation time, cooking time, macronutrients, and steps. an example of what the structure should look like is as follows: <script type=\"application/ld+json\">\n    {\n      \"@context\": \"https://schema.org/\",\n      \"@type\": \"Recipe\",\n      \"name\": \"Non-Alcoholic Piña Colada\",\n      \"image\": [\n      \"https://example.com/photos/1x1/photo.jpg\",\n      \"https://example.com/photos/4x3/photo.jpg\",\n      \"https://example.com/photos/16x9/photo.jpg\"\n      ],\n      \"author\": {\n        \"@type\": \"Person\",\n        \"name\": \"Mary Stone\"\n      },\n      \"datePublished\": \"2018-03-10\",\n      \"description\": \"This non-alcoholic pina colada is everyone's favorite!\",\n      \"recipeCuisine\": \"American\",\n      \"prepTime\": \"PT1M\",\n      \"cookTime\": \"PT2M\",\n      \"totalTime\": \"PT3M\",\n      \"keywords\": \"non-alcoholic\",\n      \"recipeYield\": \"4 servings\",\n      \"recipeCategory\": \"Drink\",\n      \"nutrition\": {\n        \"@type\": \"NutritionInformation\",\n        \"calories\": \"120 calories\"\n      },\n      \"aggregateRating\": {\n        \"@type\": \"AggregateRating\",\n        \"ratingValue\": \"5\",\n        \"ratingCount\": \"18\"\n      },\n      \"recipeIngredient\": [\n        \"400ml of pineapple juice\",\n        \"100ml cream of coconut\",\n        \"ice\"\n      ],\n      \"recipeInstructions\": [\n        {\n          \"@type\": \"HowToStep\",\n          \"name\": \"Blend\",\n          \"text\": \"Blend 400ml of pineapple juice and 100ml cream of coconut until smooth.\",\n          \"url\": \"https://example.com/non-alcoholic-pina-colada#step1\",\n          \"image\": \"https://example.com/photos/non-alcoholic-pina-colada/step1.jpg\"\n        },\n        {\n          \"@type\": \"HowToStep\",\n          \"name\": \"Fill\",\n          \"text\": \"Fill a glass with ice.\",\n          \"url\": \"https://example.com/non-alcoholic-pina-colada#step2\",\n          \"image\": \"https://example.com/photos/non-alcoholic-pina-colada/step2.jpg\"\n        },\n        {\n          \"@type\": \"HowToStep\",\n          \"name\": \"Pour\",\n          \"text\": \"Pour the pineapple juice and coconut mixture over ice.\",\n          \"url\": \"https://example.com/non-alcoholic-pina-colada#step3\",\n          \"image\": \"https://example.com/photos/non-alcoholic-pina-colada/step3.jpg\"\n        }\n      ],\n      \"video\": {\n        \"@type\": \"VideoObject\",\n        \"name\": \"How to Make a Non-Alcoholic Piña Colada\",\n        \"description\": \"This is how you make a non-alcoholic piña colada.\",\n        \"thumbnailUrl\": [\n          \"https://example.com/photos/1x1/photo.jpg\",\n          \"https://example.com/photos/4x3/photo.jpg\",\n          \"https://example.com/photos/16x9/photo.jpg\"\n         ],\n        \"contentUrl\": \"https://www.example.com/video123.mp4\",\n        \"embedUrl\": \"https://www.example.com/videoplayer?video=123\",\n        \"uploadDate\": \"2018-02-05T08:00:00+08:00\",\n        \"duration\": \"PT1M33S\",\n        \"interactionStatistic\": {\n          \"@type\": \"InteractionCounter\",\n          \"interactionType\": { \"@type\": \"WatchAction\" },\n          \"userInteractionCount\": 2347\n        },\n        \"expires\": \"2019-02-05T08:00:00+08:00\"\n       }\n    }\n    </script>\n"
    gpt_user_prompt = text

    message = [
        {"role": "system", "content": "You will only respond with structured json data matching the template layout, no comments, code blocks, backticks or emojis are present in your response."},
        {"role": "assistant", "content": gpt_assistant_prompt},
        {"role": "user", "content": gpt_user_prompt}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=message,
            temperature=0.2,
            max_tokens=2048,
            frequency_penalty=0.0,
        )

        print(response.choices[0].message.content)
        parsed_recipe = response.choices[0].message.content
        return parsed_recipe
    except Exception as e:
        print(f"Error parsing recipe with OpenAI: {e}")
        return "Error parsing recipe."

def save_recipe_as_html(recipe_data, filepath):
    # Generate HTML content from the recipe schema
    html_content = generate_html_from_schema(recipe_data)
    
    if not html_content:  # Fallback in case of an error or empty response
        html_content = "<html><body><h1>Error generating recipe content.</h1></body></html>"
    
    # Save the generated HTML content to a file
    try:
        with open(filepath, 'w') as file:
            file.write(html_content)
        print(f"HTML content successfully saved to {filepath}")
    except Exception as e:
        print(f"Failed to save HTML content. Error: {e}")

def generate_html_from_schema(recipe_schema):
    """
    Generates HTML content for a recipe schema using OpenAI's GPT.
    """
    gpt_assistant_prompt = f"Given the following recipe schema in JSON format, generate an HTML document that presents the recipe information in a structured and styled manner. The recipe schema is as follows: {json.dumps(recipe_schema)}"
    gpt_user_prompt = "Please generate the HTML."

    message = [
        {"role": "system", "content": "You will only respond with well formatted HTML, inline javascript and css, no comments, code blocks, backticks or emojis are present in your response."},
        {"role": "assistant", "content": gpt_assistant_prompt},
        {"role": "user", "content": gpt_user_prompt}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=message,
            temperature=0.2,
            max_tokens=2048,
            frequency_penalty=0.0,
        )

        # Assuming the generated HTML is directly in the response content
        generated_html = response.choices[0].message.content
        return generated_html
    except Exception as e:
        print(f"Error generating HTML from schema with OpenAI: {e}")
        return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
