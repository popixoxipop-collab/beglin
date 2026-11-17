#!/usr/bin/env python3
"""JE01 streaming SHA source attestor synthetic guard tests; no internet/model."""
import hashlib
import io
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_parquet_source_sha_audit as je


class MockReply:
    def __init__(self,payload,host="us.aws.cdn.hf.co",headers=None,status=200):
        self.buffer=io.BytesIO(payload)
        self.status=status
        self.host=host
        self.headers=headers or {"content-length":str(len(payload)),
                                 "content-encoding":"identity"}
    def read(self,n):
        return self.buffer.read(n)
    def geturl(self):
        return "https://"+self.host+"/immutable"
    def __enter__(self):
        return self
    def __exit__(self,*args):
        return False


class SourceAuditTest(unittest.TestCase):
    def setUp(self):
        self.data=b"PAR1"+b"verified-test-data-123"+b"PAR1"
        self.row=("mock","Salesforce/wikitext","b"*40,
                  "wikitext-103-raw-v1/train-00000-of-00002.parquet",len(self.data))
    def opener(self,payload=None,**kw):
        def f(req,timeout):
            self.assertEqual(req.get_method(),"GET")
            self.assertIn("/resolve/"+("b"*40)+"/",req.full_url)
            return MockReply(self.data if payload is None else payload,**kw)
        return f
    def test_bytes_sha_streaming(self):
        r=je.sha_source(self.row,opener=self.opener())
        self.assertEqual(r["file_sha256"],hashlib.sha256(self.data).hexdigest())
        self.assertEqual(r["byte_size"],len(self.data))
        self.assertTrue(r["parquet_magic_verified"])
    def test_tampered_size_denied(self):
        with self.assertRaisesRegex(ValueError,"SIZE_DRIFT|HEADER_DRIFT"):
            je.sha_source(self.row,opener=self.opener(payload=self.data[:-1]))
    def test_magic_denied(self):
        fake=b"FAKE"+self.data[4:]
        with self.assertRaisesRegex(ValueError,"MAGIC_INVALID"):
            je.sha_source(self.row,opener=self.opener(payload=fake))
    def test_external_redirect_denied(self):
        with self.assertRaisesRegex(ValueError,"UNTRUSTED_DATA_REDIRECT"):
            je.sha_source(self.row,opener=self.opener(host="untrusted.invalid"))
    def test_non_200_denied(self):
        with self.assertRaisesRegex(ValueError,"TRANSFER_NOT_RAW_200"):
            je.sha_source(self.row,opener=self.opener(status=206))
    def test_compressed_response_denied(self):
        with self.assertRaisesRegex(ValueError,"TRANSFER_NOT_RAW_200"):
            je.sha_source(self.row,opener=self.opener(headers={
                "content-length":str(len(self.data)),"content-encoding":"gzip"}))


if __name__=="__main__":
    unittest.main(verbosity=2)
