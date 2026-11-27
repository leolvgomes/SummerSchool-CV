"""Train the provided person dataset on Windows (CPU or CUDA)."""
from pathlib import Path
import argparse
import json
import multiprocessing
import shutil
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', default='auto', help='auto, cpu, or 0 for CUDA GPU')
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = ROOT / 'dataset/data.yaml'
    config.write_text(yaml.safe_dump({
        'path': str(ROOT / 'dataset'),
        'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
        'names': {0: 'person'},
    }, sort_keys=False), encoding='utf-8')
    for split in ('train', 'val', 'test'):
        images = list((ROOT / 'dataset' / split / 'images').glob('*.jpg'))
        if not images:
            raise RuntimeError(f'Missing dataset images: {split}. Extract the whole ZIP.')
        for image in images:
            label = ROOT / 'dataset' / split / 'labels' / image.with_suffix('.txt').name
            if not label.is_file():
                raise FileNotFoundError(label)
    print(f'Training device: {device}', flush=True)
    model = YOLO(str(ROOT / 'modelos/yolo12n.pt'))
    model.train(data=str(config), epochs=args.epochs, imgsz=416,
                batch=args.batch, device=device, workers=0, patience=5, freeze=10,
                project=str(ROOT / 'runs'), name='people', seed=42, plots=True)
    best = Path(model.trainer.save_dir) / 'weights/best.pt'
    target = ROOT / 'modelos/pessoas.pt'
    shutil.copy2(best, target)
    trained = YOLO(str(target))
    metrics = trained.val(data=str(config), split='test', imgsz=416,
                          device=device, workers=0, project=str(ROOT / 'runs'), name='held_out_test')
    (ROOT / 'runs/test_metrics.json').write_text(json.dumps(metrics.results_dict, indent=2), encoding='utf-8')
    print(f'Trained model saved: {target}', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
