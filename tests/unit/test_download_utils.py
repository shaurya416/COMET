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
