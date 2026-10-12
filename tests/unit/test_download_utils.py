# -*- coding: utf-8 -*-
import importlib.util
import io
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing through the `comet` package pulls in torch, which this module does not
# need, so it is loaded from its file.
_spec = importlib.util.spec_from_file_location(
    "_comet_download_utils",
    os.path.join(ROOT, "comet", "models", "download_utils.py"),
)
download_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_utils)


class TestDownloadFileMaybeExtract(unittest.TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(tempfile.mkdtemp())
        self.cache = self.tmpdir / "cache"
        # Random bytes do not compress, so truncating the archive cuts the member.
        payload = os.urandom(256 * 1024)
        self.archive = self.tmpdir / "my-model.tar.gz"
        with tarfile.open(self.archive, "w:gz") as tar:
            info = tarfile.TarInfo("my-model/checkpoints/model.ckpt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_valid_archive_is_extracted(self):
        download_utils.download_file_maybe_extract(
            self.archive.as_uri(), directory=str(self.cache)
        )
        checkpoint = self.cache / "my-model" / "checkpoints" / "model.ckpt"
        self.assertEqual(checkpoint.stat().st_size, 256 * 1024)

    def test_truncated_archive_raises(self):
        data = self.archive.read_bytes()
        self.archive.write_bytes(data[: len(data) // 2])
        with self.assertRaises(subprocess.CalledProcessError):
            download_utils.download_file_maybe_extract(
                self.archive.as_uri(), directory=str(self.cache)
            )

    def test_corrupt_archive_raises(self):
        self.archive.write_bytes(b"<Error><Code>AccessDenied</Code></Error>")
        with self.assertRaises(subprocess.CalledProcessError):
            download_utils.download_file_maybe_extract(
                self.archive.as_uri(), directory=str(self.cache)
            )


class TestDownloadModelLegacy(unittest.TestCase):
    model = "my-model"

    def setUp(self):
        self.tmpdir = pathlib.Path(tempfile.mkdtemp())
        self.cache = self.tmpdir / "cache"
        self.payload = os.urandom(256 * 1024)
        self.archive = self.tmpdir / "my-model.tar.gz"
        with tarfile.open(self.archive, "w:gz") as tar:
            info = tarfile.TarInfo("my-model/checkpoints/model.ckpt")
            info.size = len(self.payload)
            tar.addfile(info, io.BytesIO(self.payload))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def download(self):
        """Runs download_model_legacy with the archive served from disk."""

        def urlretrieve(url, filename, reporthook=None):
            shutil.copyfile(self.archive, filename)

        url = "https://example.invalid/my-model.tar.gz"
        with mock.patch.dict(
            download_utils.available_legacy_metrics, {self.model: url}
        ), mock.patch.object(
            download_utils.urllib.request, "urlretrieve", urlretrieve
        ):
            return download_utils.download_model_legacy(self.model, str(self.cache))

    def test_valid_archive_returns_checkpoint(self):
        checkpoint = pathlib.Path(self.download())
        self.assertEqual(checkpoint.name, "model.ckpt")
        self.assertEqual(checkpoint.read_bytes(), self.payload)
        self.assertFalse((self.cache / "my-model.tar.gz").exists())

    def test_failed_extraction_is_not_reused_as_cache(self):
        data = self.archive.read_bytes()
        self.archive.write_bytes(data[: len(data) // 2])
        with self.assertRaises(subprocess.CalledProcessError):
            self.download()
        # The partially extracted model must not be served by the next call.
        self.assertFalse((self.cache / self.model).exists())
        with self.assertRaises(subprocess.CalledProcessError):
            self.download()
