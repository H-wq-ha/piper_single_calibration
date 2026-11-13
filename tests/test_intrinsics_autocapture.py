import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from handeye.board import create_board, create_detector
from handeye.vision import CharucoFrontend
from intrinsics.autocapture import AutoCaptureGate, frame_quality, view_descriptor
from intrinsics.main_capture import (calibrate_camera, clear_previous_images,
                                     image_directory_lock, save_frame, save_result)


class IntrinsicsTests(unittest.TestCase):
    def setUp(self):
        self.board, _ = create_board()
        self.gray = self.board.generateImage((1100, 800), marginSize=30)
        self.image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        self.corners, self.ids, _, _ = create_detector(self.board).detectBoard(self.gray)
        self.descriptor = view_descriptor(self.board, self.corners, self.ids, (1100, 800))

    def test_restart_clears_old_photos_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, current, outside = root/'images/old', root/'images/current', root/'outside'
            for folder in [old, current, outside]:
                folder.mkdir(parents=True)
                (folder/'0000.png').write_bytes(b'photo')
            (root/'images/legacy.PNG').write_bytes(b'old photo')
            (old/'session.json').write_text('{}')
            (root/'images/.gitkeep').touch()
            (root/'calibration.yaml').write_text('saved result')
            (root/'images/external').symlink_to(outside, target_is_directory=True)
            self.assertEqual(clear_previous_images(root/'images', current), 2)
            self.assertFalse((old/'0000.png').exists())
            self.assertTrue((current/'0000.png').exists())
            self.assertTrue((outside/'0000.png').exists())
            self.assertTrue((old/'session.json').exists())
            self.assertTrue((root/'images/.gitkeep').exists())
            self.assertEqual((root/'calibration.yaml').read_text(), 'saved result')

    def test_second_instance_cannot_clear_active_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('intrinsics.main_capture.IMAGE_ROOT', Path(tmp)):
                with image_directory_lock():
                    with self.assertRaisesRegex(RuntimeError, '另一个内参程序'):
                        with image_directory_lock():
                            self.fail('Second instance acquired the image lock')
                with image_directory_lock():
                    pass

    def test_shared_board_and_quality(self):
        obj, img, ids = CharucoFrontend().detect(self.image)
        self.assertEqual(len(ids), 70)
        np.testing.assert_allclose(obj.max(axis=0), [.180, .126, 0], atol=1e-7)
        self.assertTrue(frame_quality(self.gray, self.corners, self.ids)[0])
        blurred = cv2.GaussianBlur(self.gray, (41, 41), 12)
        self.assertFalse(frame_quality(blurred, self.corners, self.ids)[0])

    def test_stability_interval_and_distinct_views(self):
        gate = AutoCaptureGate()
        self.assertFalse(gate.consider(self.corners, self.ids, self.descriptor, 0)[0])
        self.assertTrue(gate.consider(self.corners, self.ids, self.descriptor, .7)[0])
        gate.mark_saved(self.descriptor, .7)
        self.assertFalse(gate.consider(self.corners, self.ids, self.descriptor, 1)[0])
        self.assertFalse(gate.consider(self.corners, self.ids, self.descriptor, 3)[0])
        moved = self.corners + np.array([80, 0], dtype=np.float32)
        other_view = self.descriptor + [.1, 0]
        self.assertFalse(gate.consider(moved, self.ids, other_view, 4)[0])
        self.assertTrue(gate.consider(moved, self.ids, other_view, 4.7)[0])
        gate.reset_motion()
        self.assertFalse(gate.consider(moved, self.ids, other_view, 8)[0])

    def test_raw_capture_is_lossless_and_never_overwrites(self):
        _, _, quality = frame_quality(self.gray, self.corners, self.ids)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = save_frame(self.image, directory, 0, self.corners, self.ids, quality, True)
            np.testing.assert_array_equal(cv2.imread(str(path)), self.image)
            record = json.loads((directory / 'samples.jsonl').read_text())
            self.assertEqual(record['image'], '0000.png')
            self.assertEqual(len(record['ids']), 70)
            with self.assertRaises(FileExistsError):
                save_frame(self.image, directory, 0, self.corners, self.ids, quality, True)

    def test_mixed_resolution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'a.png', Path(tmp)/'b.png'
            cv2.imwrite(str(first), self.image)
            cv2.imwrite(str(second), cv2.resize(self.image, (550, 400)))
            with self.assertRaisesRegex(ValueError, '分辨率不一致'):
                calibrate_camera(self.board, [first, second])

    def test_solver_recovers_known_intrinsics_and_preserves_old_result(self):
        K = np.array([[610., 0, 320], [0, 605., 240], [0, 0, 1]])
        dist = np.array([.05, -.02, .001, -.002, .005])
        obj = self.board.getChessboardCorners()
        observations = []
        for i in range(12):
            r = np.array([-.4+.08*i, .3*np.sin(i), -.2+.04*i])
            t = np.array([-.1+.025*(i % 4), -.07+.02*(i % 3), .38+.035*i])
            points = cv2.projectPoints(obj, r, t, K, dist)[0]
            observations.append((points, np.arange(70, dtype=np.int32).reshape(-1, 1)))
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{i}.png' for i in range(12)]
            for path in paths:
                cv2.imwrite(str(path), np.zeros((480, 640, 3), np.uint8))
            with patch('intrinsics.main_capture.detect_charuco', side_effect=observations):
                result = calibrate_camera(self.board, paths)
            np.testing.assert_allclose(result['K'], K, atol=.02)
            np.testing.assert_allclose(result['dist'].flatten(), dist, atol=.002)
            self.assertEqual(len(result['per_view_rms_px']), 12)
            output = Path(tmp)/'calibration.yaml'
            output.write_text('original calibration\n')
            save_result(result, output)
            backups = list(Path(tmp).glob('calibration.backup_*.yaml'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), 'original calibration\n')
            exported = yaml.safe_load(output.read_text())
            self.assertTrue(exported['board']['legacy_pattern'])
            self.assertEqual(exported['image_count'], 12)


if __name__ == '__main__':
    unittest.main()
