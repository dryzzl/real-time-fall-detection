import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from fall_detection.cli import main
from fall_detection.contracts import FramePacket
from fall_detection.pose import (
    LetterboxTransform,
    OnnxPoseEstimator,
    PoseConfig,
    PoseError,
    decode_output,
    load_pose_config,
    preprocess_frame,
)


class FakeSession:
    def __init__(self, output=None, *, input_shape=None, output_shape=None,
                 input_type="tensor(float)", output_type="tensor(float)"):
        self.output = output
        self.input = SimpleNamespace(name="images", shape=input_shape or [1, 3, 100, 100], type=input_type)
        self.result = SimpleNamespace(name="output0", shape=output_shape or [1, 56, "anchors"], type=output_type)
        self.feed = None

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.result]

    def run(self, names, feed):
        self.feed = (names, feed)
        if isinstance(self.output, BaseException):
            raise self.output
        return [self.output]


class PoseConfigTests(unittest.TestCase):
    def test_config_file_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.toml"
            path.write_text('[pose]\nmodel_path="model.onnx"\ninput_width=320\nconfidence_threshold=0.4\n', encoding="utf-8")
            config = load_pose_config(path, input_width=640, input_height=None)
            self.assertEqual(config.model_path, Path("model.onnx"))
            self.assertEqual((config.input_width, config.input_height), (640, 640))
            self.assertEqual(config.confidence_threshold, 0.4)

    def test_invalid_and_unknown_settings_are_rejected(self):
        invalid = (
            {"model_path": Path("model.pt")}, {"input_width": 31}, {"input_height": True},
            {"keypoint_count": 0}, {"confidence_threshold": float("nan")},
            {"confidence_threshold": True}, {"iou_threshold": 1.1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PoseConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in ("[run]\nmodel_path='x.onnx'\n", "[pose]\nunknown=1\n",
                            "pose=1\n", "[pose]\nmodel_path=1\n"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_pose_config(path)

    def test_missing_model_and_runtime_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.onnx"
            with self.assertRaisesRegex(PoseError, "pose model not found"):
                OnnxPoseEstimator(PoseConfig(model_path=missing))
            model = Path(directory) / "present.onnx"
            model.touch()
            with patch("fall_detection.pose.importlib.import_module", side_effect=ImportError), self.assertRaisesRegex(PoseError, "model extra"):
                OnnxPoseEstimator(PoseConfig(model_path=model))

    def test_cli_missing_model_is_nonzero_and_does_not_claim_inference(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(["pose-check", "--model", "models/missing.onnx"])
        self.assertEqual(code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("not found", errors.getvalue())


@unittest.skipIf(np is None, "install the model extra for pose tensor tests")
class PreprocessTests(unittest.TestCase):
    def test_letterbox_shape_rgb_normalization_and_transform(self):
        frame = FramePacket(0, 0, 4, 2, bytes([10, 20, 30]) * 8)
        config = PoseConfig(input_width=32, input_height=32)
        tensor, transform = preprocess_frame(frame, config)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 8, 0], [30 / 255, 20 / 255, 10 / 255])
        np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
        self.assertEqual((transform.scale_x, transform.scale_y, transform.pad_left, transform.pad_top), (8, 8, 0, 8))
        self.assertEqual(transform.source_xy(16, 16), (2, 1))

    def test_non_square_rounding_has_separate_reversible_scales(self):
        frame = FramePacket(0, 0, 7, 3, b"\0" * 63)
        _, transform = preprocess_frame(frame, PoseConfig(input_width=100, input_height=64))
        self.assertNotEqual(transform.scale_x, transform.scale_y)
        x, y = transform.source_xy(transform.pad_left + 3 * transform.scale_x,
                                   transform.pad_top + 2 * transform.scale_y)
        self.assertAlmostEqual(x, 3)
        self.assertAlmostEqual(y, 2)


@unittest.skipIf(np is None, "install the model extra for pose decoding tests")
class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.config = PoseConfig(input_width=100, input_height=100)
        self.transform = LetterboxTransform(200, 100, 100, 100, 0.5, 0.5, 0, 25)

    def row(self, confidence=0.9, center=(50, 50), size=(40, 20), point=(50, 50, 0.8)):
        return [*center, *size, confidence, *point * 17]

    def raw(self, rows):
        return np.asarray(rows, dtype=np.float32).T[None]

    def test_decodes_boxes_keypoints_and_transposed_layout(self):
        detection = decode_output(self.raw([self.row()]), self.transform, self.config)[0]
        np.testing.assert_allclose(detection.box_xyxy, [60, 30, 140, 70])
        self.assertEqual(len(detection.keypoints), 17)
        self.assertAlmostEqual(detection.keypoints[0].x, 100)
        self.assertAlmostEqual(detection.keypoints[0].y, 50)
        self.assertAlmostEqual(detection.keypoints[0].confidence, 0.8)
        transposed = np.asarray([self.row()], dtype=np.float32)[None]
        self.assertEqual(decode_output(transposed, self.transform, self.config), (detection,))

    def test_empty_low_confidence_nonfinite_and_degenerate_rows_are_safe(self):
        empty = np.empty((1, 56, 0), dtype=np.float32)
        self.assertEqual(decode_output(empty, self.transform, self.config), ())
        nonfinite = self.row()
        nonfinite[0] = float("nan")
        rows = [self.row(confidence=0.24), self.row(confidence=1.2), nonfinite,
                self.row(size=(0, 20))]
        self.assertEqual(decode_output(self.raw(rows), self.transform, self.config), ())

    def test_clips_coordinates_scores_and_applies_nms(self):
        first = self.row(confidence=0.9, center=(50, 50), size=(200, 200), point=(-5, 200, 1.4))
        second = self.row(confidence=0.8, center=(51, 50), size=(200, 200))
        selected = decode_output(self.raw([second, first]), self.transform, self.config)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].confidence, 0.9)
        self.assertEqual(selected[0].box_xyxy, (0, 0, 199, 99))
        self.assertEqual((selected[0].keypoints[0].x, selected[0].keypoints[0].y,
                          selected[0].keypoints[0].confidence), (0, 99, 1))

    def test_wrong_output_shapes_are_rejected(self):
        for shape in ((56,), (2, 56, 10), (1, 55, 10), (1, 10, 55)):
            with self.subTest(shape=shape), self.assertRaises(PoseError):
                decode_output(np.zeros(shape, dtype=np.float32), self.transform, self.config)


@unittest.skipIf(np is None, "install the model extra for pose adapter tests")
class AdapterTests(unittest.TestCase):
    def test_session_contract_and_inference_feed(self):
        raw = np.zeros((1, 56, 0), dtype=np.float32)
        session = FakeSession(raw)
        estimator = OnnxPoseEstimator(PoseConfig(input_width=100, input_height=100), session=session)
        detections = estimator.estimate(FramePacket(0, 0, 20, 10, b"\0" * 600))
        self.assertEqual(detections, ())
        names, feed = session.feed
        self.assertEqual(names, ["output0"])
        self.assertEqual(feed["images"].shape, (1, 3, 100, 100))
        self.assertEqual(estimator.describe()["provider"], "CPUExecutionProvider")

    def test_dynamic_model_dimensions_are_supported(self):
        session = FakeSession(np.zeros((1, 56, 0), dtype=np.float32),
                              input_shape=["batch", 3, "height", "width"],
                              output_shape=["batch", 56, "anchors"])
        estimator = OnnxPoseEstimator(PoseConfig(input_width=320, input_height=192), session=session)
        self.assertEqual(estimator.describe()["input_shape"], [1, 3, 192, 320])

    def test_invalid_model_metadata_is_rejected(self):
        cases = (
            FakeSession(input_shape=[1, 1, 100, 100]),
            FakeSession(input_shape=[2, 3, 100, 100]),
            FakeSession(input_shape=[1, 3, 64, 100]),
            FakeSession(input_shape=[1, 3, 100, 64]),
            FakeSession(input_type="tensor(uint8)"),
            FakeSession(output_shape=[1, 55, 10]),
            FakeSession(output_shape=[1, 56]),
            FakeSession(output_type="tensor(double)"),
        )
        for session in cases:
            with self.subTest(shape=session.input.shape, output=session.result.shape), self.assertRaises(PoseError):
                OnnxPoseEstimator(PoseConfig(input_width=100, input_height=100), session=session)

    def test_runtime_failure_is_wrapped(self):
        session = FakeSession(RuntimeError("execution failed"))
        estimator = OnnxPoseEstimator(PoseConfig(input_width=100, input_height=100), session=session)
        with self.assertRaisesRegex(PoseError, "execution failed"):
            estimator.estimate(FramePacket(0, 0, 20, 10, b"\0" * 600))


try:
    import onnx
    import onnxruntime
except ImportError:
    onnx = onnxruntime = None


@unittest.skipIf(onnx is None or onnxruntime is None or np is None,
                 "install model and test extras for the real runtime smoke test")
class OnnxRuntimeSmokeTests(unittest.TestCase):
    def make_fixture(self, path: Path):
        helper, tensor = onnx.helper, onnx.TensorProto
        values = np.zeros((1, 56, 1), dtype=np.float32)
        values[0, 0:5, 0] = [32, 32, 20, 24, 0.9]
        for offset in range(5, 56, 3):
            values[0, offset:offset + 3, 0] = [32, 32, 0.8]
        constant = helper.make_node(
            "Constant", inputs=[], outputs=["output0"],
            value=helper.make_tensor("poses", tensor.FLOAT, values.shape, values.flatten()),
        )
        graph = helper.make_graph(
            [constant], "pose-contract-fixture",
            [helper.make_tensor_value_info("images", tensor.FLOAT, [1, 3, 64, 64])],
            [helper.make_tensor_value_info("output0", tensor.FLOAT, [1, 56, 1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        onnx.save(model, path)

    def test_real_cpu_session_and_cli_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fixture.onnx"
            self.make_fixture(model)
            estimator = OnnxPoseEstimator(PoseConfig(model_path=model, input_width=64, input_height=64))
            detections = estimator.estimate(FramePacket(0, 0, 64, 64, b"\0" * (64 * 64 * 3)))
            self.assertEqual(len(detections), 1)
            result = subprocess.run(
                [sys.executable, "-m", "fall_detection", "pose-check", "--model", str(model),
                 "--input-width", "64", "--input-height", "64", "--smoke"],
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual((record["provider"], record["pose_count"], record["smoke_run"]),
                             ("CPUExecutionProvider", 1, True))
            self.assertEqual(record["smoke_frame"], "black_test_frame")


if __name__ == "__main__":
    unittest.main()
