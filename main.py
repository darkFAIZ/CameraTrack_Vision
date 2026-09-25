"""Draw a circle in front of the camera and play a size-based animation."""

from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SMALL_VIDEO = SCRIPT_DIR / "small_circle.mp4"
LARGE_VIDEO = SCRIPT_DIR / "large_circle.mp4"
WINDOW_NAME = "CameraTrack Vision"


def get_index_tip(hand_landmarks: object, frame_width: int, frame_height: int) -> tuple[int, int]:
	"""Convert the normalized index-finger landmark to pixel coordinates."""
	landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
	return int(landmark.x * frame_width), int(landmark.y * frame_height)


def is_drawing(hand_landmarks: object) -> bool:
	"""Draw while the index finger is extended, regardless of hand tilt."""
	landmarks = hand_landmarks.landmark
	mcp = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
	pip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
	tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

	# Compare the full 3D landmark distances so turning the hand does not make
	# an extended finger look bent just because its tip moved lower in the image.
	segment_length = np.linalg.norm(np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z]))
	finger_length = np.linalg.norm(np.array([tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z]))
	return finger_length > segment_length * 1.25


def classify_circle(points: list[tuple[int, int]], board_width: int, board_height: int) -> str | None:
	"""Classify a finished stroke using only its bounding-box diameter."""
	if len(points) < 10:
		return None

	coordinates = np.asarray(points, dtype=np.int32)
	width = int(coordinates[:, 0].max() - coordinates[:, 0].min())
	height = int(coordinates[:, 1].max() - coordinates[:, 1].min())
	diameter = max(width, height)
	if diameter < 30:
		return None

	# Use the larger bounding-box dimension as the drawn diameter. This accepts
	# imperfect or slightly oval circles without checking closure or roundness.
	size_cutoff = min(board_width, board_height) * 0.35
	return "small" if diameter < size_cutoff else "large"


def fit_video_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
	"""Scale to fill the display area, then crop the excess from the center."""
	frame_height, frame_width = frame.shape[:2]
	scale = max(width / frame_width, height / frame_height)
	resized_width = round(frame_width * scale)
	resized_height = round(frame_height * scale)
	resized = cv2.resize(frame, (resized_width, resized_height))
	x_start = (resized_width - width) // 2
	y_start = (resized_height - height) // 2
	return resized[y_start:y_start + height, x_start:x_start + width]


def play_video(video_path: Path) -> None:
	"""Play one animation and return to the camera when it ends."""
	if not video_path.exists():
		print(f"Video not found: {video_path}")
		return

	video = cv2.VideoCapture(str(video_path))
	if not video.isOpened():
		print(f"Could not open video: {video_path}")
		return

	while True:
		success, frame = video.read()
		if not success:
			break
		cv2.imshow(WINDOW_NAME, fit_video_frame(frame, 640, 480))
		if cv2.waitKey(30) & 0xFF == ord("q"):
			video.release()
			raise KeyboardInterrupt
	video.release()


def main() -> None:
	# MediaPipe 0.10.30+ removed the legacy Solutions API used by this script.
	if not hasattr(mp, "solutions"):
		raise RuntimeError(
			"This script needs MediaPipe's legacy Hands API, which is missing from this Python environment. "
			"Use Python 3.12 and install the pinned dependencies with: "
			"python -m pip install -r requirements.txt"
		)

	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		camera.release()
		raise RuntimeError("Could not open the camera. Close other camera apps, check camera permissions, and try camera index 0.")

	hands_module = mp.solutions.hands
	board_width, board_height = 640, 480
	stroke: list[tuple[int, int]] = []
	last_drawing_time = 0.0

	with hands_module.Hands(
		static_image_mode=False,
		max_num_hands=1,
		min_detection_confidence=0.6,
		min_tracking_confidence=0.55,
	) as hands:
		try:
			while True:
				success, frame = camera.read()
				if not success:
					break

				frame = cv2.flip(frame, 1)
				frame = cv2.resize(frame, (board_width, board_height))
				results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				currently_drawing = False

				if results.multi_hand_landmarks:
					hand = results.multi_hand_landmarks[0]
					tip = get_index_tip(hand, board_width, board_height)
					if is_drawing(hand) and 0 <= tip[0] < board_width and 0 <= tip[1] < board_height:
						currently_drawing = True
						if not stroke or np.linalg.norm(np.array(tip) - np.array(stroke[-1])) > 3:
							stroke.append(tip)
						last_drawing_time = time.monotonic()
						cv2.circle(frame, tip, 8, (0, 255, 255), -1)

				# Allow brief tracking gaps while the hand turns around the circle.
				# Open or incomplete strokes are rejected by classify_circle.
				if stroke and not currently_drawing and time.monotonic() - last_drawing_time > 1.0:
					size = classify_circle(stroke, board_width, board_height)
					if size:
						play_video(SMALL_VIDEO if size == "small" else LARGE_VIDEO)
					else:
						print("The stroke was not recognized as a circle. Try again.")
					stroke.clear()

				if len(stroke) > 1:
					cv2.polylines(frame, [np.array(stroke)], False, (0, 255, 0), 4)

				cv2.putText(frame, "Raise index finger and draw a circle", (15, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
				cv2.putText(frame, "Lower finger to play | C: clear | Q: quit", (15, 58),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
				cv2.imshow(WINDOW_NAME, frame)

				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				if key == ord("c"):
					stroke.clear()
		finally:
			camera.release()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
