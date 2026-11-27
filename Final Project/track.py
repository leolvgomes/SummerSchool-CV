"""Detecção e ByteTrack com duas fontes: vídeo e webcam."""
from pathlib import Path
import argparse
import json
import time
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
TRAIL_LENGTH = 40
TRAIL_COLORS = [(80, 255, 80), (0, 200, 255), (255, 180, 60), (210, 100, 255), (255, 220, 80)]

def executar(fonte='video',mostrar=True,confianca=None,camera_index=0,camera_backend="opencv"):
    if confianca is None:confianca=0.10 if fonte=='webcam' else 0.35
    tracker_config = ROOT / ('botsort_camera.yaml' if fonte=='webcam' else 'bytetrack_people.yaml')
    weights=ROOT/'modelos/pessoas.pt'
    if not weights.is_file():raise FileNotFoundError('Modelo ausente: mantenha modelos/pessoas.pt dentro desta pasta.')
    source=camera_index if fonte=='webcam' else str(ROOT/'videos/videoTracking.mp4')
    if fonte=='webcam':
        cap=cv2.VideoCapture(source, cv2.CAP_DSHOW) if __import__('sys').platform=='win32' else cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            cap=cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError('Não foi possível abrir a câmera. No Windows, habilite o acesso a camera para aplicativos da area de trabalho em Configuracoes > Privacidade e seguranca > Camera. Feche outros programas que usam a camera ou tente outro indice.' if fonte=='webcam' else 'Não foi possível abrir o vídeo.')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0 or fps>120:fps=30
    output=ROOT/'resultados';output.mkdir(exist_ok=True)
    print(f'Detector: {weights.name} | Tracker: {tracker_config.name}',flush=True)
    writer=None;model=YOLO(str(weights));seen=set();frames=0;log=[];inference_seconds=0.0
    window_name = 'People tracking - ' + fonte + ' | Q to quit'
    trails = {}
    last_seen = {}
    try:
        if mostrar:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        while True:
            ok,frame=cap.read()
            if not ok:break
            tick=time.perf_counter()
            r=model.track(frame,persist=True,tracker=str(tracker_config),classes=[0],conf=confianca,imgsz=640 if fonte=='webcam' else 416,iou=0.45,device='cpu',verbose=False)[0]
            elapsed=time.perf_counter()-tick
            inference_seconds+=elapsed
            annotated=r.plot()
            ids=[] if r.boxes.id is None else r.boxes.id.int().tolist()
            seen.update(ids);frames+=1
            # Um rastro curto por ID; uma interrupção longa inicia outro segmento.
            if ids:
                for tid, (cx, cy) in zip(ids, r.boxes.xywh[:, :2].cpu().tolist()):
                    if tid not in trails or frames - last_seen.get(tid, frames) > 5:
                        trails[tid] = deque(maxlen=TRAIL_LENGTH)
                    trails[tid].append((round(cx), round(cy)))
                    last_seen[tid] = frames
                    if len(trails[tid]) > 1:
                        points = np.asarray(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(annotated, [points], False, TRAIL_COLORS[(tid - 1) % len(TRAIL_COLORS)], 3, cv2.LINE_AA)
            for tid in list(last_seen):
                if frames - last_seen[tid] > TRAIL_LENGTH:
                    last_seen.pop(tid)
                    trails.pop(tid, None)
            if fonte=='video':
                log.append({'frame':frames,'ids':ids,'boxes':r.boxes.xyxy.tolist(),'confidence':r.boxes.conf.tolist()})
            cv2.putText(annotated,f'Pessoas: {len(r.boxes)} | IDs: {len(seen)} | Frame: {frames}',(15,30),0,.65,(0,255,255),2)
            if fonte=='video':
                if writer is None:
                    h,w=frame.shape[:2];writer=cv2.VideoWriter(str(output/'video_com_trajetorias.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    if not writer.isOpened():raise RuntimeError('Falha ao salvar vídeo.')
                writer.write(annotated)
            if mostrar:
                cv2.imshow(window_name, annotated)
                delay=max(1,round(1000/fps-elapsed*1000)) if fonte=='video' else 1
                if cv2.waitKey(delay) & 255 in (ord('q'), 27):break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)<1:break
            if frames%100==0:print('Frames processados:',frames,flush=True)
    finally:
        cap.release()
        if writer:writer.release()
        if mostrar:cv2.destroyAllWindows()
    if fonte=='video':
        (output/'rastreamento.json').write_text(json.dumps(log))
        (output/'resumo_rastreamento.json').write_text(json.dumps({'frames':frames,'ids':sorted(seen),'inference_fps':frames/inference_seconds if inference_seconds else 0,'confidence':confianca},indent=2))
    print(f'Concluído: {frames} frames; {len(seen)} IDs distintos (não equivalem necessariamente a pessoas únicas).',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fonte',choices=['video','webcam'],default='video');p.add_argument('--camera-index',type=int,default=0);p.add_argument('--sem-janela',action='store_true');p.add_argument('--confianca',type=float,default=None);a=p.parse_args()
    executar(a.fonte,not a.sem_janela,a.confianca,a.camera_index)
