
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from extract_faces import load_detector, crop_face
from train_face_cremad import CLASSES, build_model
from train_audio_cremad import Net, SR, MAX_LEN, load_wave


def get_face_probs(video_path, model, device, n_frames=10):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise RuntimeError("Не удалось определить количество кадров")

    indices = np.linspace(
        0.2 * total, 0.8 * total, n_frames
    ).astype(int)

    detector = load_detector()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    probabilities = []

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()

            if not ok:
                continue

            face = crop_face(detector, frame)

            if face is None:
                continue

            image = Image.fromarray(
                cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            )

            x = tf(image).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.autocast(
                    device_type=device,
                    enabled=(device == "cuda")
                ):
                    logits = model(x)

                probs = torch.softmax(logits.float(), dim=1)
                probabilities.append(probs.cpu().numpy()[0])
    finally:
        cap.release()

    if not probabilities:
        return None, 0

    return np.mean(probabilities, axis=0), len(probabilities)


def get_audio_path(video_path, temp_dir):
    # Для оригинальных CREMA-D-видео используем готовый WAV.
    dataset_wav = Path("data/cremad/AudioWAV") / (
        video_path.stem + ".wav"
    )

    if dataset_wav.exists():
        return dataset_wav

    # Проверяем, есть ли аудиодорожка.
    if subprocess.run(
        ["ffprobe", "-version"],
        capture_output=True
    ).returncode == 0:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path)
            ],
            capture_output=True,
            text=True
        )

        if probe.returncode == 0 and not probe.stdout.strip():
            print("В видео нет звука. Используем только лицо.")
            return None

    # Извлекаем аудио.
    output_wav = Path(temp_dir) / "extracted_audio.wav"

    command = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1",
        "-ar", str(SR),
        str(output_wav)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if (
        result.returncode != 0
        or not output_wav.exists()
        or output_wav.stat().st_size == 0
    ):
        if "does not contain any stream" in result.stderr.lower():
            print("В видео нет звука. Используем только лицо.")
            return None

        print("Не удалось извлечь аудио. Используем только лицо.")
        return None

    return output_wav


def get_audio_probs(audio_path, model, device):
    if audio_path is None:
        return None

    y = load_wave(audio_path)

    if len(y) == 0:
        return None

    if len(y) > MAX_LEN:
        start = (len(y) - MAX_LEN) // 2
        y = y[start:start + MAX_LEN]

    elif len(y) < MAX_LEN:
        repeats = int(np.ceil(MAX_LEN / len(y)))
        y = np.tile(y, repeats)[:MAX_LEN]

    x = torch.from_numpy(
        y.copy()
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device,
            enabled=(device == "cuda")
        ):
            logits = model(x)

        probs = torch.softmax(logits.float(), dim=1)

    return probs.cpu().numpy()[0]


def print_probs(title, probs):
    print(f"\n{title}")

    if probs is None:
        print("Нет данных для анализа.")
        return

    order = np.argsort(probs)[::-1]

    for i in order:
        print(f"  {CLASSES[i]:10s}: {probs[i]:.3f}")

    print("  Предсказание:", CLASSES[int(np.argmax(probs))])


def main():
    parser = argparse.ArgumentParser(
        description="CREMA-D video emotion prediction"
    )

    parser.add_argument("video", help="Путь к видео")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--w-face", type=float, default=0.5)

    parser.add_argument(
        "--face-weights",
        default="outputs/face_cremad_frozen.pth"
    )

    parser.add_argument(
        "--audio-weights",
        default="outputs/audio_cremad_w2v_frozen.pth"
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise SystemExit(f"Видео не найдено: {video_path}")

    if args.frames < 1:
        raise SystemExit("--frames должен быть не меньше 1")

    if not 0 <= args.w_face <= 1:
        raise SystemExit("--w-face должен быть от 0 до 1")

    if not Path(args.face_weights).exists():
        raise SystemExit(
            f"Не найдены веса модели: {args.face_weights}"
        )

    if not Path(args.audio_weights).exists():
        raise SystemExit(
            f"Не найдены веса модели: {args.audio_weights}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Устройство:", device)
    print("Видео:", video_path)

    # Загружаем модель лица.
    face_model = build_model(
        "frozen", len(CLASSES)
    ).to(device)

    face_state = torch.load(
        args.face_weights,
        map_location=device,
        weights_only=True
    )

    face_model.load_state_dict(face_state)
    face_model.eval()

    # Загружаем модель аудио.
    audio_model = Net(
        len(CLASSES), 0
    ).to(device)

    audio_state = torch.load(
        args.audio_weights,
        map_location=device,
        weights_only=True
    )

    audio_model.load_state_dict(audio_state)
    audio_model.eval()

    # Анализируем лицо.
    face_probs, face_count = get_face_probs(
        video_path,
        face_model,
        device,
        args.frames
    )

    # Анализируем аудио, если оно существует.
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = get_audio_path(video_path, temp_dir)

        if audio_path is not None:
            audio_probs = get_audio_probs(
                audio_path,
                audio_model,
                device
            )
        else:
            audio_probs = None

    print_probs(
        f"=== FACE CREMA-D (кадров с лицом: {face_count}) ===",
        face_probs
    )

    if audio_probs is not None:
        print_probs("=== AUDIO CREMA-D ===", audio_probs)

    # Объединяем результаты, только если доступны обе модели.
    if face_probs is not None and audio_probs is not None:
        final = (
            args.w_face * face_probs
            + (1 - args.w_face) * audio_probs
        )
        print_probs("=== FUSED CREMA-D ===", final)

    elif face_probs is not None:
        final = face_probs
        print("\nИспользуется только распознавание лица.")
        print_probs("=== FACE-ONLY RESULT ===", final)

    elif audio_probs is not None:
        final = audio_probs
        print("\nИспользуется только распознавание голоса.")
        print_probs("=== AUDIO-ONLY RESULT ===", final)

    else:
        raise SystemExit(
            "Не удалось обнаружить ни лицо, ни аудио."
        )

    print(
        "\nИтоговая эмоция:",
        CLASSES[int(np.argmax(final))].upper()
    )

    # Для исходных записей CREMA-D показываем ожидаемую метку.
    parts = video_path.stem.split("_")

    code_to_emotion = {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fearful",
        "DIS": "disgust",
    }

    expected = (
        code_to_emotion.get(parts[2])
        if len(parts) >= 3
        else None
    )

    if expected:
        print("Ожидаемая эмоция по имени файла:", expected)
        print("(Это метка записи, а не результат модели.)")


if __name__ == "__main__":
    main()
