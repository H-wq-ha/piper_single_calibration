"""Shared definition of the checked-in Calib.io ChArUco target (metres)."""
import cv2


SQUARES = (11, 8)  # X/Y order in the landscape PDF, not its 8x11 filename.
SQUARE_M = .018
MARKER_M = .013
DICTIONARY = "DICT_5X5_100"
LEGACY_PATTERN = True


def board_spec():
    return {"type": "charuco", "squares": list(SQUARES), "square_m": SQUARE_M,
            "marker_m": MARKER_M, "dictionary": DICTIONARY,
            "legacy_pattern": LEGACY_PATTERN}


def create_board(squares=SQUARES, square_m=SQUARE_M, marker_m=MARKER_M,
                 legacy_pattern=LEGACY_PATTERN):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    board = cv2.aruco.CharucoBoard(tuple(squares), square_m, marker_m, dictionary)
    board.setLegacyPattern(legacy_pattern)
    return board, dictionary


def create_detector(board):
    params = cv2.aruco.CharucoParameters()
    params.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(board, params, cv2.aruco.DetectorParameters())
