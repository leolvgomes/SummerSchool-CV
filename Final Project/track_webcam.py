"""Run a Windows webcam with the trained model."""
import argparse
from track import executar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    executar("webcam", camera_index=args.camera_index)
