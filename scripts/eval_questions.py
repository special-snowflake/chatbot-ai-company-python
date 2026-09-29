"""Labeled evaluation set for the NOVAHAUS catalog assistant.

Every expectation here is cited to the source line it comes from, so a failure
means a real regression rather than a drifting assumption.

Fields per case:
    id        stable identifier
    category  failure class being measured
    query     the user question, verbatim
    expect    "answer" | "refuse" | "escalate"
    all       substrings that must all appear (case-insensitive)
    any       substrings of which min_any must appear
    min_any   how many of `any` must appear (default 1)
    none      substrings that must not appear
    note      where the expected answer comes from
"""

from __future__ import annotations

from typing import Any, Dict, List

C = dict

CASES: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ #
    # single-intent: one question, one unambiguous answer
    # ------------------------------------------------------------------ #
    C(id="single-delivery-java", category="single-intent",
      query="How long does delivery take within Java?",
      expect="answer", all=["2-5 business days"],
      note="FAQ 8, line 203: 'Typical delivery is 2-5 business days.'"),
    C(id="single-return-window", category="single-intent",
      query="How long is the return window?",
      expect="answer", all=["7 calendar days"],
      note="FAQ 9, line 223: 'returned within 7 calendar days'."),
    C(id="single-glow-warranty", category="single-intent",
      query="How long is the Glow Bulb warranty?",
      expect="answer", all=["2-year warranty"],
      note="FAQ 9, line 235: 'Glow lighting products carry a 2-year warranty.'"),
    C(id="single-plug-mini-load", category="single-intent",
      query="What is the maximum load of Connect Plug Mini?",
      expect="answer", all=["10a"],
      note="FAQ 5, line 125: 'rated for loads up to 10A'."),
    C(id="single-hubpro-zigbee", category="single-intent",
      query="How many Zigbee devices can Home Hub Pro support?",
      expect="answer", all=["100"],
      note="FAQ 6: 'Home Hub Pro supports up to 100 Zigbee devices.'"),
    C(id="single-campan-storage", category="single-intent",
      query="What storage does Secure Cam Pan support?",
      expect="answer", all=["256gb"],
      note="FAQ 7: 'supports microSD cards up to 256GB'."),
    C(id="single-sense-temp", category="single-intent",
      query="Which product measures temperature and humidity?",
      expect="answer", all=["sense temp"],
      note="FAQ 3: 'Sense Temp measures temperature and humidity.'"),
    C(id="single-glow-socket", category="single-intent",
      query="What socket does Glow Bulb A19 use?",
      expect="answer", all=["e27"],
      note="FAQ 4: 'Glow Bulb A19 uses an E27 socket.'"),
    C(id="single-doorbell-ip", category="single-intent",
      query="Is Secure Doorbell weather resistant?",
      expect="answer", all=["ip54"],
      note="FAQ 7: 'Secure Doorbell is rated IP54'."),
    C(id="single-based", category="single-intent",
      query="Where is NOVAHAUS based?",
      expect="answer", all=["jakarta"],
      note="FAQ 1: 'NOVAHAUS is headquartered in Jakarta, Indonesia.'"),

    C(id="single-founded", category="single-intent",
  query="When was NOVAHAUS founded?",
  expect="answer", all=["2021"],
  note="FAQ 1, line 33: 'NOVAHAUS was founded in 2021.'"),
C(id="single-plug-warranty", category="single-intent",
  query="How long is the Connect Plug warranty?",
  expect="answer", all=["1-year warranty"],
  note="FAQ 9, line 238: 'carry a 1-year warranty'."),

# ------------------------------------------------------------------ #
# paraphrase: the same question, worded as a human would ask it.
# The original pipeline failed 11 of 15 reworded groups at 0.58, so
# these are the cases that prove the gate is no longer the bottleneck.
# ------------------------------------------------------------------ #
C(id="para-shipment-days", category="paraphrase",
  query="how many days does shipment usually takes?",
  expect="answer", any=["1-3 business days", "2-5 business days",
                        "3-8 business days"], min_any=1,
  note="FAQ 8. Region unspecified, so any cited window is defensible."),
C(id="para-bulb-warranty", category="paraphrase",
  query="what's the length of the warranty on glow bulbs?",
  expect="answer", all=["2-year"],
  note="FAQ 9, line 235, reworded."),
C(id="para-send-back", category="paraphrase",
  query="how many days do I have to send something back?",
  expect="answer", all=["7 calendar days"],
  note="FAQ 9, line 223, reworded."),
C(id="para-airquality", category="paraphrase",
  query="which sense device tracks the air quality?",
  expect="answer", all=["sense air"],
  note="FAQ 3: 'Sense Air measures PM2.5, TVOC, CO2-equivalent...'."),
C(id="para-hubpro-capacity", category="paraphrase",
  query="what's the top zigbee capacity on the pro hub?",
  expect="answer", all=["100"],
  note="FAQ 6, reworded."),
C(id="para-doorbell-rain", category="paraphrase",
  query="can the doorbell sit outside when it rains?",
  expect="answer", all=["ip54"],
  note="FAQ 7: IP54, sheltered outdoor use."),
C(id="para-cheapest", category="paraphrase",
  query="how much is the cheapest item you sell?",
  expect="answer", all=["129"],
  note="FAQ 2, line 53: 'Glow Bulb A19 at Rp129,000'."),
C(id="para-international", category="paraphrase",
  query="do you deliver to countries outside indonesia?",
  expect="answer", any=["indonesia"], min_any=1,
  note="FAQ 8, line 218: 'service is limited to Indonesia'."),

# ------------------------------------------------------------------ #
# tie: an ambiguous question whose true answer is several variants.
# The original returned ONE arbitrarily; these require all of them.
# ------------------------------------------------------------------ #
C(id="tie-delivery-generic", category="tie",
  query="how many days does delivery take?",
  expect="answer",
  all=["1-3 business days", "2-5 business days", "3-8 business days"],
  note="FAQ 8 lists three regions (lines 200/203/206). No region named, "
       "so the complete answer is all three."),
C(id="tie-delivery-times", category="tie",
  query="what are your delivery times?",
  expect="answer",
  all=["1-3 business days", "2-5 business days", "3-8 business days"],
  note="Same three-region set as above."),
C(id="tie-cheapest-product", category="tie",
  query="what is the cheapest product?",
  expect="answer", all=["129"],
  note="FAQ 2, line 53. Must not answer the separate 'cheapest smart "
       "plug' node (Rp149,000) for a question that asks about products."),

    # ------------------------------------------------------------------ #
    # compound: two questions joined by "and". The original answered one
    # of them and silently dropped the other (measured: 5 of 5 sampled
    # compound questions lost half their intent).
    # ------------------------------------------------------------------ #
    C(id="compound-shipping-cost-and-time", category="compound",
      query="what are the shipping costs and how long does delivery take?",
      expect="answer", all=["business days", "does not contain"],
      note="The corpus has NO shipping-cost content at all (verified: zero "
           "matches for cost/ongkir/fee anywhere in source/). Delivery time "
           "is answerable, cost is not, so the reply must state both."),
    C(id="compound-lighting-and-cost", category="compound",
      query="what smart lighting products do you sell and what do they cost?",
      expect="answer",
      any=["glow bulb a19", "glow bulb rgb", "glow strip 2m", "glow ceiling s"],
      min_any=2,
      note="Catalog: 4 Glow products (Rp129,000 / Rp179,000 / Rp249,000 / "
           "Rp399,000). A single-product reply is the old failure."),
    C(id="compound-bulb-detail-and-price", category="compound",
      query="tell me about the Glow Bulb A19 and how much it costs",
      expect="answer", all=["e27"], any=["129"], min_any=1,
      note="FAQ 4 socket + catalog price Rp129,000."),
    C(id="compound-delivery-and-return", category="compound",
      query="how long is delivery and how long is the return window?",
      expect="answer", all=["business days", "7 calendar days"],
      note="FAQ 8 delivery plus FAQ 9 return window."),
    C(id="compound-install-and-neutral", category="compound",
      query="can I install a smart switch myself and does it need a "
            "neutral wire?",
      expect="answer", all=["electrician"], any=["neutral"], min_any=1,
      note="FAQ 5, line 140: qualified electrician. FAQ 5, line 134: "
           "Connect Switch 1G requires a neutral wire."),
    C(id="compound-warranty-and-price", category="compound",
      query="which product has the longest warranty and what is the price?",
      expect="answer", any=["2-year"], min_any=1,
      note="FAQ 9: Glow lighting and Secure Cam carry 2 years; prices are "
           "in the catalog. The warranty half must survive."),
    C(id="compound-price-two-products", category="compound",
      query="what is the price of the Glow Ceiling S and the Connect Plug "
            "Mini?",
      expect="answer", any=["399", "149"], min_any=2,
      note="Catalog: Rp399,000 and Rp149,000. Both halves required."),
    C(id="compound-hub-and-sensors", category="compound",
      query="which hub should I buy and which sensors need a hub?",
      expect="answer", any=["home hub"], min_any=1,
      note="FAQ 6 hub comparison plus FAQ 6 'sensors require a hub'."),

    # ------------------------------------------------------------------ #
    # comparison: an explicit request to contrast two products. A
    # selector-style answerer cannot do this at all -- it can only echo
    # one node -- so both sides must appear.
    # ------------------------------------------------------------------ #
    C(id="compare-hubs", category="comparison",
      query="compare the Home Hub Mini and the Home Hub Pro",
      expect="answer", any=["home hub mini", "home hub pro"], min_any=2,
      note="FAQ 6: Mini = 50 devices, Pro = 100 devices, Rp399,000 vs "
           "Rp699,000."),
    C(id="compare-cams", category="comparison",
      query="what is the difference between Secure Cam Indoor and Secure "
            "Cam Pan?",
      expect="answer", any=["secure cam indoor", "secure cam pan"], min_any=2,
      note="FAQ 7: Indoor = 128GB storage; Pan = 256GB, 355 pan / 100 tilt."),
    C(id="compare-bulbs", category="comparison",
      query="Glow Bulb A19 vs Glow Bulb RGB",
      expect="answer", any=["glow bulb a19", "glow bulb rgb"], min_any=2,
      note="Catalog: A19 = E27, Rp129,000; RGB = colour control, "
           "Rp179,000."),

    # ------------------------------------------------------------------ #
    # safety: the highest-stakes class. The original refused these outright
    # ("My smart switch is sparking and there is a burning smell" scored
    # below the gate), which is the worst possible outcome for a hazard
    # report. The catalog states its own escalation policy, so the correct
    # answer is that policy, quoted.
    # ------------------------------------------------------------------ #
    C(id="safety-sparking-switch", category="safety",
      query="My smart switch is sparking and there is a burning smell",
      expect="escalate", all=["burning smell"],
      note="FAQ 13, line 313 lists 'burning smell' as an escalation trigger; "
           "line 317 forbids bypass advice. Safety must never be gated on "
           "similarity score."),
    C(id="safety-smoking-plug", category="safety",
      query="my smart plug is smoking and it smells like it is melting",
      expect="escalate", all=["smoke", "melting"],
      note="FAQ 13, line 313: 'smoke, melting'."),
    C(id="safety-exposed-wiring", category="safety",
      query="there is exposed wiring near my smart switch",
      expect="escalate", all=["exposed wiring"],
      note="FAQ 13, line 313: 'exposed wiring'."),
    C(id="safety-wet-bulb-sparks", category="safety",
      query="my bulb got wet and now it sparks when I turn it on",
      expect="escalate", all=["qualified electrician"],
      note="Catalog 4, line 183: fixed electrical work by a qualified "
           "electrician. Checks the safety pool spans both source files."),

    # ------------------------------------------------------------------ #
    # out-of-scope: must stay a clean refusal. These already passed before
    # and must keep passing -- the risk of a more permissive gate is that
    # unrelated questions start getting answered.
    # ------------------------------------------------------------------ #
    C(id="oos-unrelated-day", category="out-of-scope",
      query="what did you do yesterday?",
      expect="refuse",
      note="Nothing in the corpus. Baseline refused this correctly."),
    C(id="oos-president", category="out-of-scope",
      query="who is the president of Indonesia?",
      expect="refuse",
      note="Outside the catalog."),
    C(id="oos-iphone-price", category="out-of-scope",
      query="what is the price of the iPhone 17?",
      expect="refuse",
      note="Outside the catalog."),
    C(id="oos-poem", category="out-of-scope",
      query="can you write me a poem about lamps?",
      expect="refuse",
      note="Outside the catalog."),
    C(id="oos-shipping-cost", category="out-of-scope",
      query="what are your shipping costs?",
      expect="refuse",
      note="No shipping-cost content exists. The honest answer is that the "
           "catalog does not cover it. KNOWN RISK: may be answered with a "
           "delivery-time node, which is off-target rather than harmful."),
    C(id="oos-iphone-reset", category="out-of-scope",
      query="how do I reset my iPhone?",
      expect="refuse",
      note="Outside the catalog."),
    C(id="oos-joke", category="out-of-scope",
      query="tell me a joke",
      expect="refuse",
      note="Outside the catalog."),

    # ------------------------------------------------------------------ #
    # conversational: handled before retrieval. "hi there" is included
    # because the ported greeting table matched "hi" but not "hi there",
    # which then fell through to the catalog and got refused.
    # ------------------------------------------------------------------ #
    C(id="conv-hello", category="conversational",
      query="hello",
      expect="answer",
      note="Greeting table. Baseline already handled this."),
    C(id="conv-hi-there", category="conversational",
      query="hi there",
      expect="answer",
      note="The gap: 'hi' matched, 'hi there' did not, so the baseline "
           "refused a plain greeting."),
    C(id="conv-terima-kasih", category="conversational",
      query="terima kasih",
      expect="answer",
      note="Indonesian thanks, in the ported alias table."),
    C(id="conv-greeting-then-question", category="conversational",
      query="hey there, how long does delivery take within Java?",
      expect="answer", all=["2-5 business days"],
      note="Greeting prefix stripped, question still answered from FAQ 8."),

    # ------------------------------------------------------------------ #
    # budget: the original filtered by price, then handed the candidates
    # to a selector which returned one of them. Every product at or below
    # the budget is a valid answer, so all of them must be listed.
    # ------------------------------------------------------------------ #
    C(id="budget-under-150k", category="budget",
      query="what products can I get for under Rp150,000?",
      expect="answer", any=["glow bulb a19", "connect plug mini"], min_any=2,
      none=["secure doorbell", "secure cam pan"],
      note="Catalog: only these two are at or below Rp150,000 "
           "(Rp129,000 and Rp149,000)."),
    C(id="budget-under-200k", category="budget",
      query="anything cheap under Rp200,000?",
      expect="answer",
      any=["glow bulb a19", "glow bulb rgb", "connect plug mini",
           "connect switch 1g", "sense door", "sense motion", "sense temp"],
      min_any=4,
      note="7 products are at or below Rp200,000. Requiring 4 of 7 tests "
           "that the list is genuinely a list."),
    C(id="budget-under-400k", category="budget",
      query="what do you sell for under Rp400,000?",
      expect="answer",
      any=["glow strip 2m", "glow ceiling s", "home hub mini",
           "connect switch 2g", "connect plug power"],
      min_any=3,
      note="12 products qualify at or below Rp400,000."),
    C(id="budget-under-100k", category="budget",
      query="is there anything under Rp100,000?",
      expect="answer", all=["129"], any=["no novahaus product"], min_any=1,
      note="Nothing is below Rp100,000 (cheapest is Rp129,000). The honest "
           "reply names the actual floor rather than picking an arbitrary "
           "node."),
    C(id="budget-under-1m", category="budget",
      query="what do you have for under Rp1,000,000?",
      expect="answer", all=["and 5 more"],
      note="All 17 products qualify; the listing caps at 12 and must say so "
           "rather than silently dropping 5."),
]

# --------------------------------------------------------------------------- #
# Convenience views
# --------------------------------------------------------------------------- #


def by_category() -> Dict[str, List[Dict[str, Any]]]:
    """Group the cases by category, preserving definition order."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for case in CASES:
        grouped.setdefault(case["category"], []).append(case)
    return grouped


CATEGORIES: List[str] = list(by_category())
