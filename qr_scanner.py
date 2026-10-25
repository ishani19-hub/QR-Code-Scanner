import time
import re
import webbrowser
from typing import List

import cv2
import pyperclip  # if this fails, run: pip install pyperclip


def is_url(text: str) -> bool:
    # Simple URL check; good enough for our use
    pattern = re.compile(r'^(https?://)?([\w.-]+\.[a-zA-Z]{2,})(/.*)?$')
    return bool(pattern.match(text.strip()))


def draw_polygon(frame, pts):
    """Draw a polygon given points of shape (N, 2)"""
    import numpy as np
    pts = pts.astype(int).reshape(-1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)


def overlay_text(frame, text, org=(20, 40)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    # 1) Start camera
    cap = cv2.VideoCapture(0)  # change to 1 if needed
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different index (0/1) or check permissions.")

    # 🔹 Step 5 fix: Set lower resolution (faster processing)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2) Create OpenCV QR detector
    detector = cv2.QRCodeDetector()

    # 3) To avoid spamming actions, remember recently-seen codes
    last_seen = {}  # text -> timestamp
    COOLDOWN_SEC = 3.0

    print("QR Scanner running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        decoded_texts: List[str] = []
        points = None

        # 4) Try multi-code decode (OpenCV >=4.5). Fallback to single if not available.
        try:
            retval, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)
            if retval:
                decoded_texts = [t for t in decoded_info if t]
        except Exception:
            # Some OpenCV builds don’t have detectAndDecodeMulti
            data, points = detector.detectAndDecode(frame)
            if data:
                decoded_texts = [data]

        # 5) Draw boxes if points are found
        if points is not None:
            try:
                # points can be (N, 4, 2) or a list of arrays—handle both
                if hasattr(points, "shape"):
                    for p in points:
                        draw_polygon(frame, p)
                else:
                    for p in points:
                        draw_polygon(frame, p)
            except Exception:
                pass  # be lenient on shapes

        # 6) Show results & take actions
        y = 40
        now = time.time()
        for text in decoded_texts:
            # Draw text overlay
            overlay_text(frame, f"Decoded: {text[:60]}", org=(20, y))
            y += 30

            # Throttle repeated triggers
            if now - last_seen.get(text, 0) < COOLDOWN_SEC:
                continue
            last_seen[text] = now

            # Beep (console bell) to notify
            print('\a')
            print(f"[QR] {text}")

            # Copy to clipboard
            try:
                pyperclip.copy(text)
                print("→ Copied to clipboard")
            except Exception:
                print("→ Clipboard copy skipped (pyperclip unavailable)")

            # Auto-open URLs
            if is_url(text):
                if not text.startswith("http://") and not text.startswith("https://"):
                    text = "https://" + text
                try:
                    webbrowser.open(text)
                    print("→ Opened in browser")
                except Exception:
                    print("→ Could not open browser")

        cv2.imshow("AI QR Code Scanner - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
