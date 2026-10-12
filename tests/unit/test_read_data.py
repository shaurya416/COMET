# -*- coding: utf-8 -*-
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _StubModule(types.ModuleType):
    """Module whose missing attributes are mocks, so annotations and imports of
    torch / transformers / comet internals resolve without those packages."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return mock.MagicMock(name="{}.{}".format(self.__name__, name))


def load_model_module(relpath):
    """Imports comet/models/<relpath> with its torch-based dependencies replaced.

    The CSV readers only use pandas, so they are tested without torch.
    """
    import pandas  # noqa: F401

    names = [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "transformers",
        "transformers.optimization",
        "comet",
        "comet.models",
        "comet.models.base",
        "comet.models.metrics",
        "comet.models.utils",
        "comet.modules",
        "comet.models.regression",
        "comet.models.regression.regression_metric",
    ]
    stand_ins = {name: _StubModule(name) for name in names}
    # Base classes must be real classes for the model classes to be defined.
    stand_ins["comet.models.base"].CometModel = object
    stand_ins["comet.models.regression.regression_metric"].RegressionMetric = object

    saved = {key: sys.modules.get(key) for key in stand_ins}
    sys.modules.update(stand_ins)
    try:
        spec = importlib.util.spec_from_file_location(
            "_comet_model_" + relpath.replace("/", "_")[:-3],
            os.path.join(ROOT, "comet", "models", relpath),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


class ReadDataTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write_csv(self, lines):
        path = os.path.join(self.tmpdir, "data.csv")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("".join(line + "\n" for line in lines))
        return path


class TestRegressionMetricReadData(ReadDataTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = load_model_module("regression/regression_metric.py").RegressionMetric

    def test_empty_and_na_like_segments_are_kept_as_text(self):
        path = self.write_csv(
            [
                "src,mt,ref,score",
                "Hallo,,Hello,0.25",
                "NA,None,null,0.5",
                "N/A,nan,NULL,0.75",
            ]
        )
        expected = [
            ("Hallo", "", "Hello", 0.25),
            ("NA", "None", "null", 0.5),
            ("N/A", "nan", "NULL", 0.75),
        ]
        for reader in (self.cls.read_training_data, self.cls.read_validation_data):
            records = reader(None, path)
            self.assertEqual(
                [(r["src"], r["mt"], r["ref"], r["score"]) for r in records], expected
            )

    def test_na_like_system_name_is_kept(self):
        path = self.write_csv(["src,mt,ref,score,system", "a,b,c,0.5,NA"])
        records = self.cls.read_validation_data(None, path)
        self.assertEqual(records[0]["system"], "NA")

    def test_missing_score_is_rejected(self):
        path = self.write_csv(["src,mt,ref,score", "a,b,c,", "d,e,f,0.5"])
        with self.assertRaises(ValueError):
            self.cls.read_training_data(None, path)


class TestReferencelessReadData(ReadDataTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = load_model_module(
            "regression/referenceless.py"
        ).ReferencelessRegression

    def test_empty_and_na_like_segments_are_kept_as_text(self):
        path = self.write_csv(["src,mt,score", "NA,,0.5", "None,null,0.25"])
        expected = [("NA", "", 0.5), ("None", "null", 0.25)]
        for reader in (self.cls.read_training_data, self.cls.read_validation_data):
            records = reader(None, path)
            self.assertEqual([(r["src"], r["mt"], r["score"]) for r in records], expected)


class TestRankingMetricReadData(ReadDataTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = load_model_module("ranking/ranking_metric.py").RankingMetric

    def test_empty_and_na_like_segments_are_kept_as_text(self):
        path = self.write_csv(["src,pos,neg,ref", "NA,None,,N/A"])
        records = self.cls.read_training_data(None, path)
        self.assertEqual(
            records, [{"src": "NA", "pos": "None", "neg": "", "ref": "N/A"}]
        )


class TestUnifiedMetricReadData(ReadDataTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cls = load_model_module("multitask/unified_metric.py").UnifiedMetric

    def test_empty_and_na_like_segments_are_kept_as_text(self):
        model = types.SimpleNamespace(
            hparams=types.SimpleNamespace(input_segments=["mt", "src", "ref"])
        )
        path = self.write_csv(["src,mt,ref,score,system", "NA,,null,0.5,None"])
        records = self.cls.read_training_data(model, path)
        self.assertEqual(
            records, [{"mt": "", "src": "NA", "ref": "null", "score": 0.5}]
        )
        records = self.cls.read_validation_data(model, path)
        self.assertEqual(
            records,
            [{"mt": "", "src": "NA", "ref": "null", "system": "None", "score": 0.5}],
        )
