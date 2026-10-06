import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.cli import main
from fall_detection.contracts import FallState
from fall_detection.features import PoseFeatureSample, TemporalFeatureWindow
from fall_detection.training import (
    CLASS_NAMES,
    DATASET_SCHEMA_VERSION,
    MODEL_CONTRACT_VERSION,
    TemporalSequence,
    TrainingConfig,
    TrainingError,
    TrainingPlan,
    build_subject_split,
    encode_feature_window,
    export_temporal_model,
    feature_names,
    load_sequence_dataset,
    load_training_config,
    parse_sequence_record,
    prepare_training_plan,
    run_training_smoke,
    verify_temporal_onnx_contract,
    write_split_manifest,
)


def matrix(config, *, center_y=0.4):
    rows = []
    for index in range(config.sequence_length):
        row = [
            index * 0.1, 1.0, 0.9, 0.5, center_y,
            0.2, 0.4, 0.5, 0.0, 0.0,
        ]
        row.extend([0.0, 0.0] * config.keypoint_count)
        row.extend([1.0] * config.keypoint_count)
        rows.append(row)
    return rows


def record(config, index=0, *, subject=None, label="normal"):
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "sequence_id": f"sequence-{index}",
        "subject_id": subject or f"subject-{index}",
        "session_id": f"session-{index}",
        "authorization_id": "authorized-fixture",
        "label": label,
        "features": matrix(config, center_y=0.3 + index * 0.001),
    }


def sequences(config, count=9):
    return tuple(
        parse_sequence_record(
            record(config, index, label=CLASS_NAMES[index % len(CLASS_NAMES)]), config
        )
        for index in range(count)
    )


class TrainingConfigTests(unittest.TestCase):
    def test_defaults_names_and_file_overrides(self):
        config = TrainingConfig(sequence_length=3, keypoint_count=2)
        self.assertEqual(config.feature_width, 16)
        self.assertEqual(len(feature_names(2)), 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.toml"
            path.write_text(
                "[training]\ndataset_path='private/data.jsonl'\n"
                "model_path='private/model.onnx'\nsequence_length=4\nkeypoint_count=3\n",
                encoding="utf-8",
            )
            loaded = load_training_config(path, sequence_length=5)
            self.assertEqual(loaded.dataset_path, Path("private/data.jsonl"))
            self.assertEqual(loaded.model_path, Path("private/model.onnx"))
            self.assertEqual((loaded.sequence_length, loaded.keypoint_count), (5, 3))

    def test_invalid_settings_and_documents_are_rejected(self):
        invalid = (
            {"sequence_length": 1}, {"sequence_length": True},
            {"keypoint_count": 0}, {"train_fraction": 0},
            {"validation_fraction": float("nan")}, {"test_fraction": 0.2},
            {"split_seed": ""}, {"max_sequences": True},
            {"dataset_path": "not-a-path-object"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(TrainingError):
                TrainingConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in (
                "[train]\nsequence_length=3\n",
                "[training]\nunknown=1\n",
                "training=2\n",
                "[training]\n[other]\nx=1\n",
                "[training]\ndataset_path=3\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(TrainingError):
                    load_training_config(path)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(sequence_length=3, keypoint_count=2)

    def test_valid_record_round_trips_with_fixed_feature_layout(self):
        sequence = parse_sequence_record(record(self.config), self.config)
        self.assertEqual(sequence.label, FallState.NORMAL)
        self.assertEqual(len(sequence.features), 3)
        self.assertEqual(len(sequence.features[0]), self.config.feature_width)
        self.assertEqual(parse_sequence_record(sequence.as_record(), self.config), sequence)
        json.dumps(sequence.as_record(), allow_nan=False)

    def test_unknown_label_fields_and_bad_metadata_are_rejected(self):
        cases = []
        unknown_label = record(self.config)
        unknown_label["label"] = "unknown"
        cases.append(unknown_label)
        extra = record(self.config)
        extra["extra"] = True
        cases.append(extra)
        bad_schema = record(self.config)
        bad_schema["schema_version"] = 2
        cases.append(bad_schema)
        missing_auth = record(self.config)
        missing_auth["authorization_id"] = ""
        cases.append(missing_auth)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(TrainingError):
                parse_sequence_record(value, self.config)

    def test_matrix_shape_timing_masks_and_missing_rows_are_strict(self):
        cases = []
        short = record(self.config)
        short["features"] = short["features"][:-1]
        cases.append(short)
        wrong_width = record(self.config)
        wrong_width["features"][0].pop()
        cases.append(wrong_width)
        backward = record(self.config)
        backward["features"][2][0] = 0.05
        cases.append(backward)
        missing_with_values = record(self.config)
        missing_with_values["features"][1][1] = 0
        cases.append(missing_with_values)
        masked_with_coordinates = record(self.config)
        masked_with_coordinates["features"][0][14] = 0
        masked_with_coordinates["features"][0][10] = 0.2
        cases.append(masked_with_coordinates)
        nonfinite = record(self.config)
        nonfinite["features"][0][2] = float("nan")
        cases.append(nonfinite)
        for value in cases:
            with self.subTest(), self.assertRaises(TrainingError):
                parse_sequence_record(value, self.config)

    def test_loader_rejects_blank_duplicate_and_nonfinite_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            valid = json.dumps(record(self.config), allow_nan=False)
            for text in (
                valid + "\n\n",
                valid + "\n" + valid + "\n",
                valid.replace("0.9", "NaN", 1) + "\n",
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text[-20:]), self.assertRaises(TrainingError):
                    load_sequence_dataset(path, self.config)

    def test_loader_enforces_max_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            values = [record(self.config, 0), record(self.config, 1)]
            path.write_text("".join(json.dumps(value) + "\n" for value in values),
                            encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "max_sequences"):
                load_sequence_dataset(path, replace(self.config, max_sequences=1))

    def test_runtime_window_encoder_preserves_missingness(self):
        observed = PoseFeatureSample(
            1, 0, 5.0, True, 0.9, (0.5, 0.4), (0.2, 0.4), 0.5,
            None, ((0.1, -0.1), (None, None)), (True, False),
        )
        missing = PoseFeatureSample(
            1, 1, 5.1, False, None, None, None, None, None,
            ((None, None), (None, None)), (False, False),
        )
        recovered = PoseFeatureSample(
            1, 2, 5.2, True, 0.8, (0.5, 0.5), (0.2, 0.4), 0.5,
            (0.0, 1.0), ((0.2, -0.1), (0.1, 0.3)), (True, True),
        )
        window = TemporalFeatureWindow(1, (observed, missing, recovered), 2, True)
        encoded = encode_feature_window(window, self.config)
        self.assertEqual(encoded[0][0], 0)
        self.assertAlmostEqual(encoded[2][0], 0.2)
        self.assertEqual(encoded[1][1:], (0.0,) * (self.config.feature_width - 1))
        self.assertEqual(encoded[0][14:], (1.0, 0.0))

    def test_runtime_window_encoder_rejects_wrong_length_or_keypoint_count(self):
        sample = PoseFeatureSample(
            1, 0, 0.0, False, None, None, None, None, None,
            ((None, None),), (False,),
        )
        with self.assertRaises(TrainingError):
            encode_feature_window(TemporalFeatureWindow(1, (sample,), 0, False), self.config)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(sequence_length=3, keypoint_count=2)

    def test_split_is_deterministic_order_independent_and_subject_separated(self):
        values = sequences(self.config, 9)
        first = build_subject_split(values, self.config)
        second = build_subject_split(tuple(reversed(values)), self.config)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.sequence_counts.values()), 9)
        self.assertTrue(all(value > 0 for value in first.sequence_counts.values()))
        by_subject = {}
        for assignment in first.assignments:
            by_subject.setdefault(assignment.subject_id, set()).add(assignment.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_subject.values()))

    def test_all_sessions_for_a_subject_stay_together(self):
        raw = [
            parse_sequence_record(record(self.config, 0, subject="same"), self.config),
            parse_sequence_record(record(self.config, 1, subject="same"), self.config),
        ]
        raw.extend(
            parse_sequence_record(record(self.config, index), self.config)
            for index in range(2, 6)
        )
        manifest = build_subject_split(tuple(raw), self.config)
        same_splits = {item.split for item in manifest.assignments if item.subject_id == "same"}
        self.assertEqual(len(same_splits), 1)

    def test_three_subject_minimum_and_duplicate_sequence_ids_are_rejected(self):
        values = sequences(self.config, 2)
        with self.assertRaisesRegex(TrainingError, "three subjects"):
            build_subject_split(values, self.config)
        duplicate = (values[0], replace(values[1], sequence_id=values[0].sequence_id),
                     parse_sequence_record(record(self.config, 2), self.config))
        with self.assertRaisesRegex(TrainingError, "unique"):
            build_subject_split(duplicate, self.config)

    def test_manifest_records_schema_hash_layout_and_never_overwrites(self):
        manifest = build_subject_split(sequences(self.config), self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "split.json"
            write_split_manifest(path, manifest, self.config)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_sha256"], manifest.dataset_sha256)
            self.assertEqual(payload["feature_width"], self.config.feature_width)
            self.assertEqual(payload["class_names"], list(CLASS_NAMES))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_split_manifest(path, manifest, self.config)
            self.assertEqual(path.read_bytes(), original)


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(sequence_length=3, keypoint_count=2)

    def _ready_dataset(self, path):
        values = list(sequences(self.config, 12))
        placeholder = build_subject_split(values, self.config)
        train_ids = [item.sequence_id for item in placeholder.assignments if item.split == "train"]
        label_by_id = {identifier: CLASS_NAMES[index % 3]
                       for index, identifier in enumerate(train_ids)}
        values = [replace(value, label=FallState(label_by_id.get(value.sequence_id, "normal")))
                  for value in values]
        path.write_text("".join(json.dumps(value.as_record()) + "\n" for value in values),
                        encoding="utf-8")

    def test_missing_dataset_is_an_explicit_nontraining_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"
            plan = prepare_training_plan(self.config, path)
            self.assertFalse(plan.ready)
            self.assertEqual(plan.blockers, ("authorized_labeled_dataset_missing",))
            self.assertFalse(plan.as_record()["training_performed"])
            self.assertFalse(plan.as_record()["model_artifact_available"])

    def test_valid_complete_training_partition_is_ready_for_an_injected_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            self._ready_dataset(path)
            plan = prepare_training_plan(self.config, path)
            self.assertTrue(plan.ready, plan.blockers)
            self.assertEqual(sum(plan.manifest.sequence_counts.values()), 12)
            self.assertEqual(plan.as_record()["status"], "ready_for_training_backend")

    def test_missing_training_label_blocks_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            values = [parse_sequence_record(record(self.config, index), self.config)
                      for index in range(9)]
            path.write_text("".join(json.dumps(value.as_record()) + "\n" for value in values),
                            encoding="utf-8")
            plan = prepare_training_plan(self.config, path)
            self.assertFalse(plan.ready)
            self.assertIn("falling", plan.blockers[0])
            self.assertIn("fallen", plan.blockers[0])


try:
    import numpy as np
    import onnx
    import onnxruntime
except ImportError:
    np = onnx = onnxruntime = None


@unittest.skipIf(onnx is None or onnxruntime is None or np is None,
                 "install model and test extras for ONNX export contract tests")
class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(sequence_length=3, keypoint_count=2)
        values = list(sequences(self.config, 9))
        placeholder = build_subject_split(values, self.config)
        train_ids = [item.sequence_id for item in placeholder.assignments if item.split == "train"]
        label_by_id = {
            identifier: CLASS_NAMES[index % len(CLASS_NAMES)]
            for index, identifier in enumerate(train_ids)
        }
        values = [
            replace(value, label=FallState(label_by_id.get(value.sequence_id, "normal")))
            for value in values
        ]
        self.plan = TrainingPlan(
            self.config, Path("fixture.jsonl"), tuple(values),
            build_subject_split(values, self.config), (),
        )
        self.assertTrue(self.plan.ready)

    def make_model(self, path, *, width=None, metadata=True):
        helper, tensor = onnx.helper, onnx.TensorProto
        width = width or self.config.feature_width
        values = np.zeros((1, 3), dtype=np.float32)
        constant = helper.make_node(
            "Constant", inputs=[], outputs=["logits"],
            value=helper.make_tensor("scores", tensor.FLOAT, values.shape, values.flatten()),
        )
        graph = helper.make_graph(
            [constant], "temporal-contract-fixture",
            [helper.make_tensor_value_info(
                "features", tensor.FLOAT, [1, self.config.sequence_length, width]
            )],
            [helper.make_tensor_value_info("logits", tensor.FLOAT, [1, 3])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        if metadata:
            onnx.helper.set_model_props(model, {
                "model_contract_version": str(MODEL_CONTRACT_VERSION),
                "dataset_schema_version": str(DATASET_SCHEMA_VERSION),
                "class_names": ",".join(CLASS_NAMES),
            })
        onnx.save(model, path)

    def test_real_runtime_verifier_checks_metadata_shapes_and_finite_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.onnx"
            self.make_model(path)
            contract = verify_temporal_onnx_contract(path, self.config)
            self.assertEqual(contract.class_names, CLASS_NAMES)
            self.assertEqual(contract.input_shape, (1, 3, 16))
            self.assertEqual(len(contract.sha256), 64)

    def test_wrong_width_or_missing_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.onnx"
            self.make_model(path, width=self.config.feature_width + 1)
            with self.assertRaisesRegex(TrainingError, "input shape"):
                verify_temporal_onnx_contract(path, self.config)
            self.make_model(path, metadata=False)
            with self.assertRaisesRegex(TrainingError, "metadata"):
                verify_temporal_onnx_contract(path, self.config)

    def test_export_interface_verifies_then_publishes_without_overwrite(self):
        owner = self

        class Exporter:
            def train_and_export(self, plan, output_path):
                owner.assertIs(plan, owner.plan)
                owner.make_model(output_path)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models" / "temporal.onnx"
            contract = export_temporal_model(self.plan, Exporter(), path)
            self.assertEqual(contract.model_path, path)
            self.assertTrue(path.is_file())
            original = path.read_bytes()
            with self.assertRaisesRegex(TrainingError, "already exists"):
                export_temporal_model(self.plan, Exporter(), path)
            self.assertEqual(path.read_bytes(), original)

    def test_blocked_plan_never_calls_exporter(self):
        class Exporter:
            called = False
            def train_and_export(self, plan, output_path):
                self.called = True

        exporter = Exporter()
        blocked = replace(self.plan, blockers=("missing",))
        with self.assertRaisesRegex(TrainingError, "blocked"):
            export_temporal_model(blocked, exporter)
        self.assertFalse(exporter.called)


class TrainingSmokeTests(unittest.TestCase):
    def test_dependency_free_smoke_has_no_training_metrics_or_artifact(self):
        result = run_training_smoke(TrainingConfig(sequence_length=3, keypoint_count=2))
        self.assertEqual(result["records_validated"], 9)
        self.assertFalse(result["subject_leakage"])
        self.assertFalse(result["training_performed"])
        self.assertFalse(result["model_artifact_available"])
        self.assertNotIn("accuracy", json.dumps(result).lower())

    def test_cli_smoke_and_missing_dataset_status_are_honest(self):
        smoke = io.StringIO()
        with contextlib.redirect_stdout(smoke):
            self.assertEqual(main(["training-check", "--smoke"]), 0)
        self.assertFalse(json.loads(smoke.getvalue())["model_artifact_available"])
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "training.toml"
            config.write_text(
                "[training]\ndataset_path='missing.jsonl'\n"
                "sequence_length=3\nkeypoint_count=2\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["training-check", "--config", str(config)]), 0)
            record_value = json.loads(output.getvalue())
            self.assertEqual(record_value["status"], "blocked")
            self.assertEqual(record_value["blockers"], ["authorized_labeled_dataset_missing"])


if __name__ == "__main__":
    unittest.main()
