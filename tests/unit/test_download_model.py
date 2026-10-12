# -*- coding: utf-8 -*-
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "comet", "models")


class _StubModule(types.ModuleType):
    """Module whose missing attributes are mocks, so the model classes imported by
    comet/models/__init__.py resolve without torch."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return mock.MagicMock(name="{}.{}".format(self.__name__, name))


def load_models_package():
    """Imports comet/models/__init__.py with its torch-based modules replaced.

    download_model only needs the Hub client and download_utils, so it is tested
    without torch. Returns the package and its (real) download_utils module.
    """
    import yaml  # noqa: F401

    def load(name, path, package=False):
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[MODELS] if package else None
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    stubbed = [
        "torch",
        "huggingface_hub",
        "comet",
        "comet.models.base",
        "comet.models.multitask",
        "comet.models.multitask.unified_metric",
        "comet.models.multitask.xcomet_metric",
        "comet.models.ranking",
        "comet.models.ranking.ranking_metric",
        "comet.models.regression",
        "comet.models.regression.referenceless",
        "comet.models.regression.regression_metric",
    ]
    loaded = ["comet.models", "comet.models.download_utils"]
    saved = {key: sys.modules.get(key) for key in stubbed + loaded}
    sys.modules.update({name: _StubModule(name) for name in stubbed})
    try:
        download_utils = load(
            "comet.models.download_utils", os.path.join(MODELS, "download_utils.py")
        )
        models = load("comet.models", os.path.join(MODELS, "__init__.py"), True)
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
    return models, download_utils


class TestDownloadModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.download_utils = load_models_package()

    def setUp(self):
        self.cache = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.cache)

    def download(self, model, hub_error, urlretrieve=None):
        """Runs download_model with the Hub raising `hub_error`."""

        def snapshot_download(**kwargs):
            raise hub_error

        def no_network(*args, **kwargs):
            raise AssertionError("unexpected download")

        url = "https://example.invalid/legacy-model.tar.gz"
        with mock.patch.object(
            self.models, "snapshot_download", snapshot_download
        ), mock.patch.dict(
            self.download_utils.available_legacy_metrics, {"legacy-model": url}
        ), mock.patch.object(
            self.download_utils.urllib.request,
            "urlretrieve",
            urlretrieve or no_network,
        ):
            return self.models.download_model(model, saving_directory=str(self.cache))

    def test_unknown_model_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.download("no-such-model", RuntimeError("Repository Not Found"))

    def test_hub_error_is_kept_for_a_hub_model(self):
        hub_error = RuntimeError(
            "Access to model Unbabel/gated-model is restricted. "
            "You must be authenticated to access it."
        )
        with self.assertRaises(KeyError) as context:
            self.download("Unbabel/gated-model", hub_error)
        self.assertIs(context.exception.__cause__, hub_error)
        self.assertIn("is restricted", str(context.exception))
        self.assertNotIn("not supported", str(context.exception))

    def test_failed_legacy_download_is_not_reported_as_unsupported(self):
        def urlretrieve(url, filename, reporthook=None):
            raise urllib.error.URLError("Connection reset by peer")

        with self.assertRaises(urllib.error.URLError):
            self.download("legacy-model", RuntimeError("Not on the Hub"), urlretrieve)

    def test_cached_legacy_folder_is_used(self):
        checkpoints = self.cache / "legacy-model" / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "model.ckpt").write_bytes(b"weights")
        checkpoint = self.download("legacy-model", RuntimeError("Not on the Hub"))
        self.assertEqual(pathlib.Path(checkpoint), checkpoints / "model.ckpt")


if __name__ == "__main__":
    unittest.main()
