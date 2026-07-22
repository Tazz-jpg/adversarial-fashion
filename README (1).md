# Adversarial Fashion Pattern Generator

A research tool for generating adversarial patterns that reduce detection confidence in YOLOv5-based object detectors. The project explores adversarial machine learning through the lens of "adversarial fashion" — patterns designed to be printed on wearable items as a demonstration of detector evasion techniques.

This is a research and educational project intended to help people understand the strengths and limitations of computer vision detection systems.

## Features

- Generate adversarial patches from random noise or from a base image
- Expectation-over-Transformation (EoT) optimization for robustness across rotation, scale, brightness, and contrast variations
- Flask web interface for interactive generation and preview
- Command-line interface for scripted/batch generation
- Built-in evaluation against the target detector to report confidence and evasion success

## How it works

The generator optimizes a patch of pixels to minimize a target detector's confidence score when the patch is rendered under a range of simulated real-world transformations (EoT). This is a standard technique in the adversarial examples literature, used to study detector robustness rather than to defeat any specific deployed system.

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies (PyTorch, torchvision, Ultralytics YOLOv5, OpenCV, Flask, etc.)

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Usage

### Web interface

```bash
python app.py
```

Then open `http://localhost:5001` in your browser. From there you can:

- Generate a pattern from random noise
- Upload a base image to seed the pattern
- Adjust the number of optimization iterations
- Preview and download the resulting pattern

### Command line

```bash
# Generate from random noise
python run.py --output pattern.png --iterations 300

# Generate from a base image
python run.py --input my_image.png --output pattern.png --iterations 300 --model yolov5s
```

**CLI options:**

| Flag | Description | Default |
|---|---|---|
| `--input` | Path to a base image (optional; omit to start from noise) | None |
| `--output` | Output filename | `pattern.png` |
| `--iterations` | Number of optimization iterations | `300` |
| `--size` | Pattern size in pixels | `224` |
| `--model` | Target YOLOv5 model variant | `yolov5s` |

## Configuration

Key settings live in `config.py`, including the target model, patch size, optimizer parameters, and the ranges used for EoT transformations (rotation, scale, brightness, contrast). Adjust these to experiment with different tradeoffs between pattern robustness and optimization time.

## Project structure

```
.
├── app.py              # Flask web application
├── run.py              # CLI entry point
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
└── adversarial/        # Core package (generator, optimizer, utils)
```

## Limitations

- Effectiveness is tied to the specific target model and its training data; patterns are not guaranteed to generalize across detector architectures or versions.
- Real-world performance (e.g., printed on fabric, viewed at varying distances/angles/lighting) will typically be weaker than in the simulated EoT evaluation.
- This is a research prototype, not a production security tool.

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss significant changes before submitting a PR.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
