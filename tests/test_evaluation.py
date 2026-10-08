import contextlib
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.cli import main
from fall_detection.contracts import FallState
from fall_detection.evaluation import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    EvaluationConfig,
    EvaluationError,
    EvaluationOutcome,
    EvaluationPlan,
    EvaluationResults,
    OnnxTemporalSequencePredictor,
    build_evaluation_report,
    evaluate_sequences,
    load_evaluation_config,
    load_split_manifest,
    prepare_evaluation_plan,
    run_evaluation,
    run_evaluation_smoke,
    summarize_results,
    validate_manifest_dataset,
    write_evaluation_report,
)
from fall_detection.training import (
    CLASS_NAMES,
    DATASET_SCHEMA_VERSION,
    MODEL_CONTRACT_VERSION,
    TemporalModelContract,
    TemporalSequence,
    TrainingConfig,
    build_subject_split,
    dataset_digest,
    write_split_manifest,
)


def sequence(config, index, label=FallState.NORMAL):
    rows = []
    for step in range(config.sequence_length):
        row = [
            step * 0.1, 1.0, 0.9, 0.5, 0.3 + index * 0.001,
            0.2, 0.4, 0.5, 0.0, 0.0,
        ]
        row.extend([0.0, 0.0] * config.keypoint_count)
        row.extend([1.0] * config.keypoint_count)
        rows.append(tuple(row))
    return TemporalSequence(
        f"sequence-{index:02d}", f"subject-{index:02d}", f"session-{index:02d}",
        f"authorized-replay-{index:02d}", label, tuple(rows),
    )


def evaluation_dataset(config, count=15, *, complete_test_labels=True):
    values = [sequence(config, index) for index in range(count)]
    placeholder = build_subject_split(values, config)
    test_ids = [item.sequence_id for item in placeholder.assignments if item.split == "test"]
    labels = tuple(FallState(name) for name in CLASS_NAMES)
    label_by_id = {
        identifier: labels[index % len(labels)] if complete_test_labels else FallState.NORMAL
        for index, identifier in enumerate(test_ids)
    }
    values = [replace(item, label=label_by_id.get(item.sequence_id, item.label)) for item in values]
    return tuple(values), build_subject_split(values, config)


def write_dataset(path, records):
    path.write_text(
        "".join(json.dumps(record.as_record(), allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def fake_contract(path, config):
    return TemporalModelContract(
        path,
        (1, config.sequence_length, config.feature_width),
        (1, len(CLASS_NAMES)),
        CLASS_NAMES,
        sha256(path.read_bytes()).hexdigest(),
    )


class StepClock:
    def __init__(self, step=0.001):
        self.value = 0.0
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class MappingPredictor:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, value):
        return self.predictions[value.sequence_id]


class EvaluationConfigTests(unittest.TestCase):
    def test_defaults_file_and_override_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.toml"
            path.write_text(
                "[evaluation]\ntraining_config_path='private/training.toml'\n"
                "split_manifest_path='private/split.json'\nwarmup_runs=2\n"
                "timed_runs=7\nfailure_case_limit=5\n",
                encoding="utf-8",
            )
            config = load_evaluation_config(path, timed_runs=3)
            self.assertEqual(config.training_config_path, Path("private/training.toml"))
            self.assertEqual(config.split_manifest_path, Path("private/split.json"))
            self.assertEqual((config.warmup_runs, config.timed_runs), (2, 3))
            self.assertEqual(config.failure_case_limit, 5)

    def test_invalid_settings_and_documents_are_rejected(self):
        for values in (
            {"warmup_runs": -1}, {"warmup_runs": True}, {"timed_runs": 0},
            {"failure_case_limit": -1}, {"training_config_path": "not-a-path"},
        ):
            with self.subTest(values=values), self.assertRaises(EvaluationError):
                EvaluationConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in (
                "[evaluate]\ntimed_runs=2\n",
                "[evaluation]\nunknown=1\n",
                "evaluation=2\n",
                "[evaluation]\n[other]\nx=1\n",
                "[evaluation]\nsplit_manifest_path=3\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(EvaluationError):
                    load_evaluation_config(path)


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.training = TrainingConfig(sequence_length=3, keypoint_count=2)
        self.records, self.manifest = evaluation_dataset(self.training)

    def test_manifest_round_trip_and_test_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            write_split_manifest(path, self.manifest, self.training)
            loaded = load_split_manifest(path, self.training)
            self.assertEqual(loaded, self.manifest)
            held_out = validate_manifest_dataset(loaded, self.records)
            self.assertEqual({item.label.value for item in held_out}, set(CLASS_NAMES))
            self.assertTrue(all(
                next(a for a in loaded.assignments if a.sequence_id == item.sequence_id).split
                == "test" for item in held_out
            ))

    def test_manifest_rejects_tampering_unknown_fields_and_bad_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            write_split_manifest(path, self.manifest, self.training)
            original = json.loads(path.read_text(encoding="utf-8"))
            cases = (
                {**original, "extra": True},
                {**original, "dataset_sha256": "bad"},
                {**original, "split_seed": "other"},
                {**original, "feature_width": self.training.feature_width + 1},
                {**original, "sequence_counts": {**original["sequence_counts"], "test": 99}},
            )
            for value in cases:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(keys=value.keys()), self.assertRaises(EvaluationError):
                    load_split_manifest(path, self.training)

    def test_dataset_fingerprint_and_assignment_metadata_are_enforced(self):
        with self.assertRaisesRegex(EvaluationError, "fingerprint"):
            validate_manifest_dataset(
                self.manifest,
                (replace(self.records[0], authorization_id="changed"), *self.records[1:]),
            )
        bad_assignment = replace(
            self.manifest.assignments[0], session_id="wrong-session",
        )
        changed = replace(
            self.manifest,
            assignments=(bad_assignment, *self.manifest.assignments[1:]),
        )
        with self.assertRaisesRegex(EvaluationError, "metadata mismatch"):
            validate_manifest_dataset(changed, self.records)


class EvaluationPlanTests(unittest.TestCase):
    def setUp(self):
        self.training = TrainingConfig(sequence_length=3, keypoint_count=2)

    def test_missing_inputs_are_explicit_nonmeasurement_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = replace(
                self.training,
                dataset_path=root / "missing-data.jsonl",
                model_path=root / "missing-model.onnx",
            )
            plan = prepare_evaluation_plan(
                EvaluationConfig(split_manifest_path=root / "missing-split.json"), training,
            )
            self.assertFalse(plan.ready)
            self.assertEqual(plan.blockers, (
                "authorized_labeled_replay_sequences_missing",
                "held_out_split_manifest_missing",
                "trained_temporal_model_missing",
            ))
            record = plan.as_record()
            self.assertFalse(record["quality_metrics_available"])
            self.assertFalse(record["real_world_quality_measured"])
            self.assertFalse(record["evaluation_performed"])

    def _ready_plan(self, directory, *, complete_test_labels=True):
        root = Path(directory)
        records, manifest = evaluation_dataset(
            self.training, complete_test_labels=complete_test_labels,
        )
        dataset_path = root / "data.jsonl"
        manifest_path = root / "split.json"
        model_path = root / "temporal.onnx"
        write_dataset(dataset_path, records)
        write_split_manifest(manifest_path, manifest, self.training)
        model_path.write_bytes(b"test-model-contract")
        training = replace(
            self.training, dataset_path=dataset_path, model_path=model_path,
        )
        config = EvaluationConfig(
            split_manifest_path=manifest_path, warmup_runs=0, timed_runs=1,
        )
        plan = prepare_evaluation_plan(
            config, training, model_verifier=fake_contract,
        )
        return plan

    def test_valid_dataset_manifest_and_model_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._ready_plan(directory)
            self.assertTrue(plan.ready, plan.blockers)
            self.assertEqual(plan.as_record()["status"], "ready_for_authorized_replay_evaluation")
            self.assertEqual(plan.manifest.dataset_sha256, dataset_digest(plan.records))
            self.assertEqual(set(plan.as_record()["test_class_support"]), set(CLASS_NAMES))
            self.assertFalse(plan.as_record()["quality_metrics_available"])

    def test_test_split_must_cover_every_supervised_class(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._ready_plan(directory, complete_test_labels=False)
            self.assertFalse(plan.ready)
            self.assertEqual(plan.blockers, ("test_split_missing_labels:falling,fallen",))
            with self.assertRaisesRegex(EvaluationError, "blocked"):
                run_evaluation(plan, predictor=MappingPredictor({}))


class EvaluationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.training = TrainingConfig(sequence_length=3, keypoint_count=2)
        labels = (
            FallState.NORMAL, FallState.NORMAL,
            FallState.FALLING, FallState.FALLING,
            FallState.FALLEN, FallState.FALLEN,
        )
        self.records = tuple(sequence(self.training, index, label) for index, label in enumerate(labels))

    def test_confusion_per_class_regression_and_latency_accounting(self):
        predictions = (
            FallState.NORMAL, FallState.UNKNOWN,
            FallState.FALLING, FallState.NORMAL,
            FallState.FALLEN, FallState.NORMAL,
        )
        predictor = MappingPredictor({
            record.sequence_id: prediction
            for record, prediction in zip(self.records, predictions)
        })
        config = EvaluationConfig(warmup_runs=1, timed_runs=2, failure_case_limit=2)
        results = evaluate_sequences(self.records, predictor, config, clock=StepClock())
        summary = summarize_results(results, config.failure_case_limit)
        self.assertEqual(summary["sequence_count"], 6)
        self.assertEqual(summary["correct_count"], 3)
        self.assertEqual(summary["unknown_prediction_count"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["coverage"], 5 / 6)
        self.assertEqual(summary["confusion_matrix"]["normal"]["unknown"], 1)
        self.assertAlmostEqual(summary["per_class"]["normal"]["precision"], 1 / 3)
        self.assertEqual(summary["regression"]["failure_count"], 3)
        self.assertEqual(len(summary["regression"]["failure_sequence_ids"]), 2)
        self.assertTrue(summary["regression"]["failure_sequence_ids_truncated"])
        self.assertEqual(summary["temporal_model_latency"]["count"], 12)
        self.assertAlmostEqual(summary["temporal_model_latency"]["median_ms"], 1.0)
        json.dumps(summary, allow_nan=False)

    def test_repeated_predictions_must_be_deterministic(self):
        class Predictor:
            calls = 0

            def predict(self, value):
                self.calls += 1
                return FallState.NORMAL if self.calls % 2 else FallState.FALLING

        with self.assertRaisesRegex(EvaluationError, "nondeterministic"):
            evaluate_sequences(
                self.records[:1], Predictor(),
                EvaluationConfig(warmup_runs=0, timed_runs=2), clock=StepClock(),
            )

    def test_invalid_predictor_clock_and_duplicate_sequences_fail_closed(self):
        class BadPredictor:
            def predict(self, value):
                return "normal"

        with self.assertRaisesRegex(EvaluationError, "FallState"):
            evaluate_sequences(
                self.records[:1], BadPredictor(),
                EvaluationConfig(warmup_runs=0, timed_runs=1), clock=StepClock(),
            )
        predictor = MappingPredictor({self.records[0].sequence_id: FallState.NORMAL})
        with self.assertRaisesRegex(EvaluationError, "ordered"):
            evaluate_sequences(
                self.records[:1], predictor,
                EvaluationConfig(warmup_runs=0, timed_runs=1),
                clock=StepClock(step=-0.001),
            )
        with self.assertRaisesRegex(EvaluationError, "unique"):
            evaluate_sequences(
                (self.records[0], self.records[0]), predictor,
                EvaluationConfig(warmup_runs=0, timed_runs=1), clock=StepClock(),
            )


class ReportAndSmokeTests(unittest.TestCase):
    def ready_plan(self, directory):
        training = TrainingConfig(sequence_length=3, keypoint_count=2)
        records, manifest = evaluation_dataset(training)
        root = Path(directory)
        dataset_path = root / "data.jsonl"
        manifest_path = root / "split.json"
        model_path = root / "model.onnx"
        write_dataset(dataset_path, records)
        write_split_manifest(manifest_path, manifest, training)
        model_path.write_bytes(b"fixture")
        training = replace(training, dataset_path=dataset_path, model_path=model_path)
        config = EvaluationConfig(
            split_manifest_path=manifest_path, warmup_runs=0, timed_runs=1,
            failure_case_limit=10,
        )
        return prepare_evaluation_plan(config, training, model_verifier=fake_contract)

    def test_report_is_fingerprinted_finite_aggregate_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.ready_plan(directory)
            predictor = MappingPredictor({
                record.sequence_id: record.label for record in plan.test_records
            })
            results = evaluate_sequences(
                plan.test_records, predictor, plan.config, clock=StepClock(),
            )
            report = build_evaluation_report(plan, results)
            self.assertEqual(report["schema_version"], EVALUATION_REPORT_SCHEMA_VERSION)
            self.assertEqual(report["dataset_sha256"], plan.manifest.dataset_sha256)
            self.assertEqual(report["model_sha256"], plan.model_contract.sha256)
            self.assertTrue(report["quality_metrics_available"])
            self.assertTrue(report["real_world_quality_measured"])
            self.assertFalse(report["model_training_provenance_verified_by_software"])
            self.assertFalse(report["clinical_validation"])
            self.assertFalse(report["fall_detection_available"])
            path = Path(directory) / "reports" / "evaluation.json"
            write_evaluation_report(path, report)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored, report)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_evaluation_report(path, report)
            self.assertEqual(path.read_bytes(), original)

    def test_report_rejects_blocked_plan_and_incomplete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.ready_plan(directory)
            blocked = replace(plan, blockers=("missing",))
            with self.assertRaisesRegex(EvaluationError, "blocked"):
                build_evaluation_report(blocked, EvaluationResults((), ()))
            outcome = EvaluationOutcome(
                plan.test_records[0].sequence_id,
                plan.test_records[0].label,
                plan.test_records[0].label,
                1.0,
            )
            with self.assertRaisesRegex(EvaluationError, "exactly"):
                build_evaluation_report(plan, EvaluationResults((outcome,), (1.0,)))

    def test_synthetic_smoke_never_publishes_quality_metrics(self):
        result = run_evaluation_smoke()
        encoded = json.dumps(result, allow_nan=False)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["cases_validated"], 6)
        self.assertEqual(result["latency_samples_validated"], 12)
        self.assertFalse(result["quality_metrics_available"])
        self.assertFalse(result["real_world_quality_measured"])
        self.assertFalse(result["report_written"])
        self.assertNotIn("accuracy", encoded.lower())
        self.assertNotIn("macro_f1", encoded.lower())

    def test_cli_smoke_and_missing_input_status_are_honest(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["evaluation-check", "--smoke"]), 0)
        self.assertFalse(json.loads(output.getvalue())["quality_metrics_available"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training.toml"
            training_path.write_text(
                "[training]\ndataset_path='missing-data.jsonl'\n"
                "model_path='missing-model.onnx'\nsequence_length=3\nkeypoint_count=2\n",
                encoding="utf-8",
            )
            evaluation_path = root / "evaluation.toml"
            evaluation_path.write_text(
                f"[evaluation]\ntraining_config_path='{training_path}'\n"
                f"split_manifest_path='{root / 'missing-split.json'}'\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([
                    "evaluation-check", "--config", str(evaluation_path),
                ]), 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value["status"], "blocked")
            self.assertEqual(len(value["blockers"]), 3)
            self.assertFalse(value["real_world_quality_measured"])


try:
    import numpy as np
    import onnx
    import onnxruntime
except ImportError:
    np = onnx = onnxruntime = None


@unittest.skipIf(onnx is None or onnxruntime is None or np is None,
                 "install model and test extras for real ONNX evaluation tests")
class OnnxEvaluationTests(unittest.TestCase):
    def make_model(self, path, config):
        helper, tensor = onnx.helper, onnx.TensorProto
        logits = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        constant = helper.make_node(
            "Constant", inputs=[], outputs=["logits"],
            value=helper.make_tensor("scores", tensor.FLOAT, logits.shape, logits.flatten()),
        )
        graph = helper.make_graph(
            [constant], "evaluation-contract-fixture",
            [helper.make_tensor_value_info(
                "features", tensor.FLOAT,
                [1, config.sequence_length, config.feature_width],
            )],
            [helper.make_tensor_value_info("logits", tensor.FLOAT, [1, 3])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        onnx.helper.set_model_props(model, {
            "model_contract_version": str(MODEL_CONTRACT_VERSION),
            "dataset_schema_version": str(DATASET_SCHEMA_VERSION),
            "class_names": ",".join(CLASS_NAMES),
        })
        onnx.save(model, path)

    def test_real_cpu_predictor_evaluation_and_cli_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = TrainingConfig(
                dataset_path=root / "data.jsonl", model_path=root / "model.onnx",
                sequence_length=3, keypoint_count=2,
            )
            records, manifest = evaluation_dataset(training)
            write_dataset(training.dataset_path, records)
            manifest_path = root / "split.json"
            write_split_manifest(manifest_path, manifest, training)
            self.make_model(training.model_path, training)
            config = EvaluationConfig(
                split_manifest_path=manifest_path, warmup_runs=1, timed_runs=2,
            )
            plan = prepare_evaluation_plan(config, training)
            self.assertTrue(plan.ready, plan.blockers)
            predictor = OnnxTemporalSequencePredictor(training.model_path, training)
            report = run_evaluation(plan, predictor=predictor)
            self.assertEqual(report["temporal_model_latency"]["count"], 6)
            self.assertEqual(report["confusion_matrix"]["fallen"]["fallen"], 1)
            self.assertTrue(report["quality_metrics_available"])
            training_config_path = root / "training.toml"
            training_config_path.write_text(
                f"[training]\ndataset_path='{training.dataset_path}'\n"
                f"model_path='{training.model_path}'\nsequence_length=3\nkeypoint_count=2\n",
                encoding="utf-8",
            )
            evaluation_config_path = root / "evaluation.toml"
            evaluation_config_path.write_text(
                f"[evaluation]\ntraining_config_path='{training_config_path}'\n"
                f"split_manifest_path='{manifest_path}'\nwarmup_runs=1\ntimed_runs=2\n",
                encoding="utf-8",
            )
            report_path = root / "report.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([
                    "evaluation-check", "--config", str(evaluation_config_path),
                    "--report-output", str(report_path),
                ]), 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value["report_output"], str(report_path))
            self.assertTrue(report_path.is_file())
            original = report_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "evaluation-check", "--config", str(evaluation_config_path),
                    "--report-output", str(report_path),
                ]), 2)
            self.assertEqual(report_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
