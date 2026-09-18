from __future__ import annotations
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tuner import load_rag, features, train, evaluate, tuned_rank, FEATURE_NAMES
DATA=ROOT/"data/tuning_data.json"
CORPUS=Path(os.environ["NALLAR_RAG_PACKAGE"])/"data/sample_corpus.json"

class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag=load_rag(); cls.svc=cls.rag.RAGService(CORPUS); cls.data=json.loads(DATA.read_text())
        cls.model=train(cls.svc,cls.data["train"]); cls.metrics=evaluate(cls.svc,cls.model,cls.data["validation"])
    def test_01_features(self):
        r=self.svc.retrieve("stdio JSON RPC MCP",1)[0]; x=features("stdio JSON RPC MCP",r); self.assertEqual(len(x),4); self.assertEqual(FEATURE_NAMES[0],"bias")
    def test_02_loss_decreases(self): self.assertGreater(self.model["loss_reduction"],0.5)
    def test_03_weights_changed(self): self.assertTrue(any(abs(x)>0.01 for x in self.model["weights"]))
    def test_04_training_examples(self): self.assertGreaterEqual(self.model["training_examples"],80)
    def test_05_no_llm_weight_tuning(self): self.assertFalse(self.model["llm_weight_fine_tuning"])
    def test_06_validation_baseline(self): self.assertGreaterEqual(self.metrics["baseline"]["top1_accuracy"],0.75)
    def test_07_validation_no_regression(self): self.assertGreaterEqual(self.metrics["tuned"]["top1_accuracy"],self.metrics["baseline"]["top1_accuracy"])
    def test_08_mrr_no_regression(self): self.assertGreaterEqual(self.metrics["tuned"]["mrr"],self.metrics["baseline"]["mrr"])
    def test_09_deterministic(self):
        a=tuned_rank(self.svc,self.model,"MCP stdio tools",4); b=tuned_rank(self.svc,self.model,"MCP stdio tools",4); self.assertEqual(a,b)
    def test_10_expected_mcp(self): self.assertEqual(tuned_rank(self.svc,self.model,"agent tool protocol stdio resources prompts",1)[0]["doc_id"],"mcp_server")
    def test_11_cli_train(self):
        with tempfile.TemporaryDirectory() as td:
            p=subprocess.run([sys.executable,str(ROOT/"tuner.py"),"--data",str(DATA),"--corpus",str(CORPUS),
                "--model-out",str(Path(td)/"model.json"),"--metrics-out",str(Path(td)/"metrics.json")],
                capture_output=True,text=True,timeout=60,env=os.environ.copy())
            self.assertEqual(p.returncode,0); self.assertTrue((Path(td)/"model.json").is_file()); self.assertTrue((Path(td)/"metrics.json").is_file())
    def test_12_no_network(self):
        t=(ROOT/"tuner.py").read_text().lower(); self.assertNotIn("import requests",t); self.assertNotIn("import socket",t)
if __name__=="__main__": unittest.main(verbosity=2)
