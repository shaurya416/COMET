# -*- coding: utf-8 -*-
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_cli(name):
    """Imports comet/cli/<name>.py with the model stack replaced by stand-ins.

    The CLIs only need torch, pytorch_lightning and the comet model API to load
    and run a model, which these tests fake, so they run without those packages.
    """
    # Import the real lightweight dependencies first: some of them (e.g. scipy)
    # inspect sys.modules["torch"] when they are imported.
    import jsonargparse  # noqa: F401
    import numpy  # noqa: F401
    import sacrebleu.utils  # noqa: F401
    import scipy.stats  # noqa: F401
    import tabulate  # noqa: F401

    torch = types.ModuleType("torch")
    torch.set_float32_matmul_precision = lambda *args, **kwargs: None
    torch.Tensor = object
    pytorch_lightning = types.ModuleType("pytorch_lightning")
    pytorch_lightning.seed_everything = lambda *args, **kwargs: None
    comet = types.ModuleType("comet")
    comet_models = types.ModuleType("comet.models")
    for module in (comet, comet_models):
        module.download_model = module.load_from_checkpoint = None
    comet_models.RegressionMetric = comet_models.CometModel = object
    comet_models_utils = types.ModuleType("comet.models.utils")
    comet_models_utils.split_sequence_into_sublists = None

    stand_ins = {
        "torch": torch,
        "pytorch_lightning": pytorch_lightning,
        "comet": comet,
        "comet.models": comet_models,
        "comet.models.utils": comet_models_utils,
    }
    saved = {key: sys.modules.get(key) for key in stand_ins}
    sys.modules.update(stand_ins)
    try:
        spec = importlib.util.spec_from_file_location(
            "_comet_cli_" + name, os.path.join(ROOT, "comet", "cli", name + ".py")
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
    return cli


class FakePrediction(dict):
    """Mimics comet.models.utils.Prediction attribute access."""

    __getattr__ = dict.__getitem__


class FakeModel:
    """Scores each segment with the length of its translation."""

    def __init__(self):
        self.scored = []
        self.retrieve_sentence_embedding = types.SimpleNamespace(
            cache_info=lambda: "CacheInfo(segments={})".format(len(self.scored))
        )

    def eval(self):
        pass

    def half(self):
        pass

    def requires_references(self):
        return True

    def set_embedding_cache(self):
        pass

    def predict(self, samples, **kwargs):
        self.scored.extend(samples)
        scores = [float(len(sample["mt"])) for sample in samples]
        return FakePrediction(scores=scores, system_score=sum(scores) / len(scores))


class CliTestCase(unittest.TestCase):
    cli_name = None
    command = None

    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli(cls.cli_name)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model = FakeModel()
        self.loaded_checkpoints = []

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write(self, filename, lines):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("".join(line + "\n" for line in lines))
        return path

    def load_from_checkpoint(self, checkpoint_path, *args, **kwargs):
        self.loaded_checkpoints.append(checkpoint_path)
        return self.model

    def run_cli(self, *args):
        """Returns (exit code, stdout, stderr) of the CLI run with `args`."""
        stdout, stderr = io.StringIO(), io.StringIO()
        exit_code = 0
        with mock.patch.object(
            self.cli, "download_model", lambda *a, **k: "model.ckpt"
        ), mock.patch.object(
            self.cli, "load_from_checkpoint", self.load_from_checkpoint
        ), mock.patch.object(
            sys, "argv", ["comet"] + list(args)
        ), contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(
            stderr
        ):
            try:
                getattr(self.cli, self.command)()
            except SystemExit as e:
                exit_code = e.code
        return exit_code, stdout.getvalue(), stderr.getvalue()


class TestScoreCommand(CliTestCase):
    cli_name = "score"
    command = "score_command"

    def test_aligned_files_are_scored(self):
        src = self.write("src.txt", ["a", "b", "c"])
        hyp = self.write("hyp.txt", ["A", "BB", "CCC"])
        ref = self.write("ref.txt", ["a", "b", "c"])
        exit_code, stdout, _ = self.run_cli("-s", src, "-t", hyp, "-r", ref)
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self.model.scored), 3)
        self.assertIn("score: 2.0000", stdout)

    def test_translation_with_missing_line_is_rejected(self):
        src = self.write("src.txt", ["a", "b", "c"])
        hyp = self.write("hyp.txt", ["A", "C"])
        ref = self.write("ref.txt", ["a", "b", "c"])
        exit_code, stdout, stderr = self.run_cli("-s", src, "-t", hyp, "-r", ref)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(self.model.scored, [])
        self.assertNotIn("score:", stdout)
        self.assertIn("hyp.txt has 2 lines but the sources have 3", stderr)

    def test_reference_with_extra_line_is_rejected(self):
        src = self.write("src.txt", ["a", "b"])
        hyp = self.write("hyp.txt", ["A", "B"])
        ref = self.write("ref.txt", ["x", "a", "b"])
        exit_code, stdout, stderr = self.run_cli("-s", src, "-t", hyp, "-r", ref)
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(self.model.scored, [])
        self.assertIn("ref.txt has 3 lines but the sources have 2", stderr)


class TestCompareCommand(CliTestCase):
    cli_name = "compare"
    command = "compare_command"

    def test_translation_with_extra_line_is_rejected(self):
        src = self.write("src.txt", ["a", "b"])
        hyp1 = self.write("hyp1.txt", ["A", "X", "B"])
        hyp2 = self.write("hyp2.txt", ["A", "B"])
        ref = self.write("ref.txt", ["a", "b"])
        exit_code, stdout, stderr = self.run_cli(
            "-s", src, "-t", hyp1, hyp2, "-r", ref
        )
        self.assertNotEqual(exit_code, 0)
        self.assertEqual(self.model.scored, [])
        self.assertNotIn("Summary", stdout)
        self.assertIn("hyp1.txt has 3 lines but the sources have 2", stderr)

    def test_print_cache_info_reports_cache_after_scoring(self):
        src = self.write("src.txt", ["a", "b"])
        hyp1 = self.write("hyp1.txt", ["A", "BB"])
        hyp2 = self.write("hyp2.txt", ["AA", "B"])
        ref = self.write("ref.txt", ["a", "b"])
        exit_code, stdout, _ = self.run_cli(
            "-s", src, "-t", hyp1, hyp2, "-r", ref, "--print_cache_info"
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(self.model.scored), 4)
        self.assertEqual(stdout.count("CacheInfo("), 1)
        self.assertIn("CacheInfo(segments=4)", stdout)

    def test_model_is_loaded_once(self):
        src = self.write("src.txt", ["a", "b"])
        hyp1 = self.write("hyp1.txt", ["A", "BB"])
        hyp2 = self.write("hyp2.txt", ["AA", "B"])
        ref = self.write("ref.txt", ["a", "b"])
        exit_code, stdout, _ = self.run_cli("-s", src, "-t", hyp1, hyp2, "-r", ref)
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.loaded_checkpoints, ["model.ckpt"])
        self.assertEqual(len(self.model.scored), 4)
        self.assertIn("Summary", stdout)


class FakeMbrModel(FakeModel):
    """MBR model on a host without CUDA: moving it to a GPU fails."""

    def cuda(self):
        raise RuntimeError("Torch not compiled with CUDA enabled")


class TestMbrCommand(CliTestCase):
    cli_name = "mbr"
    command = "mbr_command"

    def setUp(self):
        super().setUp()
        self.model = FakeMbrModel()

    def run_mbr(self, *args):
        import numpy as np

        def build_embeddings(sources, translations, model, batch_size):
            # One-dimensional "embedding": the length of each sentence.
            return (
                np.array([[len(s)] for s in sources], dtype=float),
                np.array([[len(t)] for t in translations], dtype=float),
            )

        def mbr_decoding(src_embeddings, mt_embeddings, model):
            # MBR score of each sample is its length, so the longest one wins.
            return mt_embeddings[:, :, 0]

        with mock.patch.object(
            self.cli, "build_embeddings", build_embeddings
        ), mock.patch.object(
            self.cli, "mbr_decoding", mbr_decoding
        ), mock.patch.object(
            self.cli.torch, "argmax", np.argmax, create=True
        ):
            return self.run_cli(*args)

    def test_cpu_only_run_with_zero_gpus(self):
        src = self.write("src.txt", ["a", "b"])
        samples = self.write("samples.txt", ["A", "AAA", "BB", "B"])
        output = os.path.join(self.tmpdir, "best.txt")
        exit_code, _, _ = self.run_mbr(
            "-s", src, "-t", samples, "--num_samples", "2", "-o", output, "--gpus", "0"
        )
        self.assertEqual(exit_code, 0)
        with open(output, encoding="utf-8") as fp:
            self.assertEqual(fp.read(), "AAA\nBB\n")

    def test_model_is_moved_to_gpu_by_default(self):
        src = self.write("src.txt", ["a"])
        samples = self.write("samples.txt", ["A", "AA"])
        output = os.path.join(self.tmpdir, "best.txt")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.run_mbr("-s", src, "-t", samples, "--num_samples", "2", "-o", output)
