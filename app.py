"""
Flask web application for the Adversarial Fashion pattern generator.
"""

import os
import base64
import io
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms

from adversarial import AdversarialPatternGenerator
from adversarial.utils import tensor_to_pil
import config

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

generator = None


def get_generator():
    global generator
    if generator is None:
        print("[app.py] Initialising generator...")
        generator = AdversarialPatternGenerator(
            model_name=config.YOLOV5_MODEL,
            patch_size=config.PATCH_SIZE,
            num_iterations=config.NUM_ITERATIONS,
            learning_rate=config.LEARNING_RATE,
            eot_samples=config.EOT_SAMPLES,
            rotation_range=config.ROTATION_RANGE,
            scale_range=config.SCALE_RANGE,
            brightness_range=config.BRIGHTNESS_RANGE,
            contrast_range=config.CONTRAST_RANGE
        )
    return generator


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        mode = data.get('mode', 'noise')
        base_image_data = data.get('image', None)
        iterations = data.get('iterations', config.NUM_ITERATIONS)

        gen = get_generator()

        if iterations != gen.optimizer.num_iterations:
            gen.optimizer.num_iterations = int(iterations)

        if mode == 'noise':
            pattern, filepath = gen.generate_from_noise(save=True)
        elif mode == 'image' and base_image_data:
            if ',' in base_image_data:
                base_image_data = base_image_data.split(',')[1]
            img_data = base64.b64decode(base_image_data)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((config.PATCH_SIZE, config.PATCH_SIZE)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img)

            pattern, filepath = gen.generate_from_image(img_tensor, save=True)
        else:
            return jsonify({'error': 'Invalid mode or missing image'}), 400

        pil_img = tensor_to_pil(pattern)
        buffered = io.BytesIO()
        pil_img.save(buffered, format='PNG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        result = gen.test_pattern(pattern)

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'confidence': result['confidence'],
            'evaded': result['evaded'],
            'filepath': filepath
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(config.OUTPUTS_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/status')
def status():
    try:
        gen = get_generator()
        return jsonify({
            'status': 'ready',
            'model': config.YOLOV5_MODEL,
            'device': gen.model.device,
            'patch_size': config.PATCH_SIZE
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("🛡️  Adversarial Fashion Pattern Generator")
    print("=" * 50)
    print(f"📍 Running at: http://{config.HOST}:{config.PORT}")
    print(f"📦 Target Model: {config.YOLOV5_MODEL}")
    print("=" * 50)
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)