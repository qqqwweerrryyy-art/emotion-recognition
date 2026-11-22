"""Распознавание эмоций по видео сразу ВСЕМИ моделями: старыми (RAVDESS, 7 классов)
и новыми (CREMA-D, 6 классов, без surprised).

Запуск из корня проекта:
    python src\\predict_ensemble.py "путь\\к\\видео.mp4"
    python src\\predict_ensemble.py "видео.mp4" --w_face 0.4

Как объединяется: для каждого класса берётся среднее по тем моделям, которые этот класс знают
(surprised знают только модели RAVDESS), затем вероятности нормируются; потом лицо и голос
смешиваются с весом --w_face. Файлы весов, которых нет, пропускаются.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from extract_faces import crop_face, load_detector  # noqa: E402
from train_audio import MAX_LEN, SR, Net, load_wave  # noqa: E402
from train_face import CLASSES, build_model  # noqa: E402

# порядок классов в 6-классовых моделях CREMA-D = CLASSES без surprised (как в prepare_cremad.py)
CLASSES6 = [c for c in CLASSES if c != "surprised"]
NORM = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
RMS_MIN = 0.003


def to7(p, names):
    """вектор вероятностей модели -> вектор из 7 классов + маска «класс известен модели»"""
    out, known = np.zeros(len(CLASSES)), np.zeros(len(CLASSES), bool)
    for v, n in zip(p, names):
        out[CLASSES.index(n)] = v
        known[CLASSES.index(n)] = True
    return out, known


def n_out(state):
    return [v.shape[0] for k, v in state.items() if k.endswith("weight") and v.ndim == 2
            and ("fc." in k or "head." in k)][-1]


def load_face(path, device):
    state = torch.load(path, map_location=device)
    n = n_out(state)
    m = build_model("frozen", n)
    m.load_state_dict(state)
    return m.to(device).eval(), (CLASSES if n == 7 else CLASSES6)


def load_audio(path, device):
    state = torch.load(path, map_location=device)
    n = n_out(state)
    m = Net(n, 0)
    m.load_state_dict(state)
    return m.to(device).eval(), (CLASSES if n == 7 else CLASSES6)


def face_inputs(video, n_frames=10):
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det = load_detector()
    imgs = []
    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int) if total > 0 else range(0)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        face = crop_face(det, fr)
        if face is not None:
            imgs.append(NORM(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
    cap.release()
    return torch.stack(imgs) if imgs else None


def audio_inputs(video, tmp):
    wav = Path(tmp) / "a.wav"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(SR), str(wav)], capture_output=True)
    if r.returncode != 0 or not wav.exists():
        return None, "в видео нет звука"
    raw, _ = librosa.load(str(wav), sr=SR)
    rms = float(np.sqrt(np.mean(raw ** 2))) if len(raw) else 0.0
    if rms < RMS_MIN:
        return None, f"звук почти тихий (RMS={rms:.4f})"
    y = load_wave(str(wav))
    if len(y) <= MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)), mode="wrap")
        wins = [y]
    else:
        step = 2 * SR
        wins = [y[s:s + MAX_LEN] for s in range(0, len(y) - MAX_LEN + 1, step)]
    return torch.from_numpy(np.stack(wins)), None


@torch.no_grad()
def run(model, x, device):
    out = []
    for i in range(0, len(x), 8):
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            o = model(x[i:i + 8].to(device))
        out.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(out).mean(0)


def combine(items):
    """items: [(p7, known)] -> среднее по моделям, знающим класс, с нормировкой"""
    s, c = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
    for p, k in items:
        s += p * k
        c += k
    p = s / np.maximum(c, 1)
    return p / p.sum()


def show(title, p):
    print(f"\n=== {title} ===")
    for i in np.argsort(-p):
        print(f"  {CLASSES[i]:<10}: {p[i]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--w_face", type=float, default=0.5)
    ap.add_argument("--face_weights", nargs="*",
                    default=["outputs/face_partial.pth", "outputs/face_cremad_frozen.pth"])
    ap.add_argument("--audio_weights", nargs="*",
                    default=["outputs/audio_w2v_partial.pth", "outputs/audio_cremad_w2v_frozen.pth"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}\nВидео: {args.video}")

    face_p, audio_p = [], []
    x = face_inputs(args.video)
    if x is None:
        print("\nЛицо не найдено ни в одном кадре.")
    else:
        print(f"\nКадров с лицом: {len(x)}")
        for w in args.face_weights:
            if not Path(w).exists():
                print(f"  (нет файла {w}, пропуск)")
                continue
            m, names = load_face(w, device)
            p = run(m, x, device)
            print(f"  лицо {Path(w).name}: {names[int(p.argmax())]} ({p.max():.2f})")
            face_p.append(to7(p, names))
            del m

    with tempfile.TemporaryDirectory() as tmp:
        xa, why = audio_inputs(args.video, tmp)
        if xa is None:
            print(f"\nГолос не используется: {why}.")
        else:
            print(f"\nОкон голоса: {len(xa)}")
            for w in args.audio_weights:
                if not Path(w).exists():
                    print(f"  (нет файла {w}, пропуск)")
                    continue
                m, names = load_audio(w, device)
                p = run(m, xa, device)
                print(f"  голос {Path(w).name}: {names[int(p.argmax())]} ({p.max():.2f})")
                audio_p.append(to7(p, names))
                del m

    pf = combine(face_p) if face_p else None
    pa = combine(audio_p) if audio_p else None
    if pf is not None:
        show("ЛИЦО (все модели)", pf)
    if pa is not None:
        show("ГОЛОС (все модели)", pa)
    if pf is not None and pa is not None:
        final = args.w_face * pf + (1 - args.w_face) * pa
        show(f"ИТОГ лицо+голос (вес лица {args.w_face})", final)
    else:
        final = pf if pf is not None else pa
        if final is None:
            raise SystemExit("\nНе удалось получить ни лицо, ни голос.")
        print("\n(использован только один канал)")
    print(f"\nИтоговая эмоция: {CLASSES[int(final.argmax())].upper()}")


if __name__ == "__main__":
    main()