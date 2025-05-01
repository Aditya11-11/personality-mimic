from langchain.prompts import ChatPromptTemplate
from groq import Groq
from utiles.globalllm import GroqLLM
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from utiles.utils import build_system_prompt, ImageProcessing

API_KEY = "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"
client = Groq(api_key=API_KEY)

groq_llm = GroqLLM(model="llama-3.1-8b-instant", api_key=API_KEY,temperature=0.4)

app = Flask(__name__)

@app.route("/analyze_image_prompt", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Extract user-provided form data
    name = request.form.get("name", "Unknown")
    age = request.form.get("age", "Unknown")
    relationship_status = request.form.get("relationship_status", "friend")
    tone = request.form.get("tone", "neutral")
    way_of_talking = request.form.get("way_of_talking", "normal")
    nature_type = request.form.get("nature_type", "undisclosed")

    # Save the uploaded image
    filename = secure_filename(image_file.filename)
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(file_path)

    try:
        # Get a description of the person from the image
        physical_description = ImageProcessing(file_path)

        # Build the system prompt
        system_prompt = build_system_prompt(
            name, age, relationship_status, tone, way_of_talking, nature_type, physical_description
        )

        return jsonify({
            "system_prompt": system_prompt,
            "image_analysis": physical_description
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    
    # Validate input
    if not data or "system_prompt" not in data or "human_msg" not in data:
        return jsonify({"error": "Missing required parameters"}), 400

    system_prompt = data["system_prompt"]
    human_msg = data["human_msg"]

    # Construct prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        ("human", human_msg.strip())
    ])

    # Create chain and invoke
    chain = prompt | groq_llm
    response = chain.invoke({})

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8001)