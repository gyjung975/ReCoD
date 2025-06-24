import base64
import anthropic

API_KEY = ""
client = anthropic.Anthropic(api_key=API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def one_image(image_path, image_type, prompt, config):
    image = encode_image(image_path)
    message = client.messages.create(
        model=config['model'],
        messages=[
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": image_type, "data": image}},
                {"type": "text", "text": prompt}]},
            {'role': 'assistant', 'content': 'The image shows'}
        ],
        max_tokens=config['max_new_tokens'],
        temperature=config['temperature'],
    )
    return message


def two_images(image_path, image_type, prompt, config):
    image = [encode_image(img) for img in image_path]
    message = client.messages.create(
        model=config['model'],
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": 'Image 1:'},
                {"type": "image", "source": {"type": "base64", "media_type": image_type[0], "data": image[0]}},
                {"type": "text", "text": 'Image 2:'},
                {"type": "image", "source": {"type": "base64", "media_type": image_type[1], "data": image[1]}},
                {"type": "text", "text": prompt}]},
            {'role': 'assistant', 'content': 'The first image shows'}
        ],
        max_tokens=config['max_new_tokens'],
        temperature=config['temperature'],
    )
    return message
