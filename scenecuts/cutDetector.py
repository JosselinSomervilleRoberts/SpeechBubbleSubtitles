import cv2
from scenedetect.detectors import ContentDetector


class CutDetector:

    def __init__(self, threshold=30.0):
        self.detector = ContentDetector(threshold=threshold)
        self.cuts = []

    def preprocess(frame):
        return cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    def process(self, frame_index, frame):
        frame = CutDetector.preprocess(frame)
        self.cuts += self.detector.process_frame(frame_index, frame)
