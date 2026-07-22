"""
Command-line entry point for generating adversarial patterns.
"""

import argparse
from adversarial import AdversarialPatternGenerator
from adversarial.utils import load_image


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial fashion patterns')
    parser.add_argument('--input', type=str, help='Path to base image (optional)')
    parser.add_argument('--output', type=str, default='pattern.png', help='Output filename')
    parser.add_argument('--iterations', type=int, default=300, help='Optimization iterations')
    parser.add_argument('--size', type=int, default=224, help='Pattern size')
    parser.add_argument('--model', type=str, default='yolov5s', help='Target model')

    args = parser.parse_args()

    generator = AdversarialPatternGenerator(
        model_name=args.model,
        patch_size=args.size,
        num_iterations=args.iterations
    )

    if args.input:
        print(f"[run.py] Loading base image from {args.input}")
        base_image = load_image(args.input, size=args.size)
        pattern, filepath = generator.generate_from_image(base_image, filename=args.output)
    else:
        pattern, filepath = generator.generate_from_noise(filename=args.output)

    print(f"[run.py] Done! Pattern saved to {filepath}")

    result = generator.test_pattern(pattern)
    print(f"[run.py] Test result - Confidence: {result['confidence']:.3f}")
    print(f"[run.py] Evaded: {result['evaded']}")


if __name__ == '__main__':
    main()