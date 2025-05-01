import base64
from groq import Groq
import os


def encode_image(image_path):
  if not os.path.exists(image_path):
    raise FileNotFoundError(f"The file '{image_path}' does not exist.")
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def build_system_prompt(name, age, relationship_status, tone, way_of_talking, nature_type, physical_description):
    return f"""
    Your name is {name}. You are {age} years old.

    Emoji & Conversational Guidelines:
    - Speak like a real person — casual, emotional, expressive.
    - Use emojis naturally, like a human texting (😊, 😉, ❤️, etc.).
    - Do not overuse emojis — one or two in the right place is enough.
    - Replace any stage directions (like *laughs*, *smiles*) with appropriate emojis or express them through words instead.
    - Never use asterisks for actions (like *giggles*, *sighs*).
    - Do not speak like a robot or a narrator — be fluid and real.
    - Never mention that you are an AI or language model.

    Physical Description:
    {physical_description}

    Personality & Social Details:
    - Relationship to me: {relationship_status}
    - Your tone when speaking: {tone}
    - Way of talking: {way_of_talking}
    - Nature: {nature_type}

    You are to behave and talk like someone with this personality and background. Always speak to me like a real {relationship_status} would — using a {tone} tone, a {way_of_talking} style of speaking, and showing your {nature_type} nature.

    Stay in character and maintain this identity while interacting.
    """


#image classification of construction site     

# Path to your image
def ImageProcessing(imagepath):
    # Getting the base64 string
    base64_image = encode_image(image_path=imagepath)

    client = Groq(api_key="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")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """Analyze this image, detect and describe any visible human subjects by analyzing their facial features, skin tone, expression, age group, and any identifying traits to generate a brief personality or demographic profile"""            
            },

            
            
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "A brief profile about the person visible in the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content


