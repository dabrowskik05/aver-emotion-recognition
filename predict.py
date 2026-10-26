import sys


def predict(video_path):
    """Przewidywana emocja dla jednego .mp4."""
    raise NotImplementedError


if __name__ == "__main__":
    print(predict(sys.argv[1]))
