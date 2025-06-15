from flask import Flask, request, render_template, send_file
from style_transfer import stylize_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

from PIL import Image, ImageEnhance

@app.route('/stylize', methods=['POST'])
def stylize():
    content_file = request.files['content']
    style_file = request.files['style']
    alpha = float(request.form.get('alpha', 1.0))

    # Нові параметри
    brightness = float(request.form.get('brightness', 1.0))
    contrast = float(request.form.get('contrast', 1.0))
    saturation = float(request.form.get('saturation', 1.0))

    content_path = 'content.jpg'
    style_path = 'style.jpg'
    output_path = 'static/result.jpg'

    content_file.save(content_path)
    style_file.save(style_path)

    # Стилізація (генерація стилізованого зображення)
    stylize_image(content_path, style_path, output_path, alpha=alpha)

    # Відкриваємо результат для подальшої обробки
    img = Image.open(output_path)

    # Застосовуємо корекції
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)

    # Зберігаємо фінальний результат
    img.save(output_path)

    return send_file(output_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
