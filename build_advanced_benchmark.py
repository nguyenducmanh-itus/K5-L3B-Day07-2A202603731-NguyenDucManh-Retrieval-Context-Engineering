"""Build the locked dev/test benchmark from hand-authored evidence anchors.

Gold evidence is built offline from the source PDF extraction and is not passed
to run_agent.py; the agent receives only the question and retrieved context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "data" / "advanced"
PAGES = ADV / "pages.jsonl"
OUT = ADV / "benchmark.json"

# id, split, primary group, question, expected answer, [(PDF page, exact evidence anchor)]
ROWS = [
    # Test: 24 answerable, six missing evidence; group minima match ADVANCED.md.
    ("WDR-T01","test","table","In Table 2.2, what proportion of low-income-country NSOs use an advance release calendar?","30% of low-income-country NSOs.",[(84,"NSO uses advance release calendar")]),
    ("WDR-T02","test","table","In Table 6.1, what is the high-income average score for e-commerce and e-transactions?","86.",[(227,"High-income                               73                       59                       43                          86")]),
    ("WDR-T03","test","table","At the 'Establishing fundamentals' stage in Table 6.2, what first action is recommended for safeguards and enablers?","Conduct a baseline needs assessment in both areas.",[(229,"Conduct a baseline needs assessment.")]),
    ("WDR-T04","test","table","According to Table 8.1, name two data-management actions for the Store stage.","Encrypt data and use secure servers; also back up and archive data.",[(288,"Encrypt data; use secure servers; back up and archive data.")]),
    ("WDR-T05","test","table","In Table 8.2, which two approaches are listed for a data-governance arrangement?","Centralized and decentralized approaches.",[(292,"Centralized approach: data governance agency/unit embedded in")]),
    ("WDR-T06","test","table","In Table 8.3, what is a stated purpose of data trusts?","To enable trusted sharing and use of sensitive data through enhanced accountability mechanisms.",[(298,"Enable trusted sharing")]),
    ("WDR-T07","test","image_diagram","On the John Snow map (Map 1.1), what do the black rectangles of different sizes represent?","Deaths from cholera.",[(47,"The solid black rectangles of various sizes represent deaths from cholera.")]),
    ("WDR-T08","test","image_diagram","What practical use of the Croatia poverty map does the caption of Map 2.1 identify?","It helped target antipoverty funds more effectively.",[(77,"allowed better targeting of"),(77,"antipoverty funds")]),
    ("WDR-T09","test","image_diagram","What safety outcome does Map 2.2 associate with citizen-collected data in Bogotá?","Greater safety around bike paths.",[(79,"citizen-collected data in"),(79,"safety around bike paths")]),
    ("WDR-T10","test","image_diagram","In Figure 1.3, how does the legend distinguish positive and negative impacts?","Positive impacts are green and negative impacts are red.",[(54,"Positive impacts are shown in green; negative impacts are shown in red.")]),
    ("WDR-T11","test","image_diagram","In Figure 1.4, what two levels are used to organize the data-governance layers?","National and international.",[(59,"Figure 1.4 Data governance layers at the national and international levels")]),
    ("WDR-T12","test","chart","What change in census enumeration does Figure 1.1 report between 1850 and today?","The share of people counted rose from about 1 in 10 in 1850 to 9 in 10 today.",[(46,"The share of people counted in a census grew from about 1 in 10 in 1850 to")]),
    ("WDR-T13","test","chart","According to Figure 2.4's caption, how does comparable poverty-data coverage in lower-income countries compare with other country groups?","Lower-income countries, especially those affected by fragility and conflict, have less comparable poverty data than other groups.",[(83,"Lower-income countries, especially those affected by fragility and")]),
    ("WDR-T14","test","chart","What is the main finding stated in Figure 2.7's caption about national statistical plans?","Most countries do not fully fund their national statistical plans.",[(86,"Most countries do not fully fund their national statistical plans")]),
    ("WDR-T15","test","chart","What relationship does Figure 2.9 report between NSO independence, press freedom, and statistical performance?","Greater NSO independence and freedom of the press are positively correlated with better statistical performance.",[(89,"Greater NSO independence and freedom of the press are positively correlated with")]),
    ("WDR-T16","test","chart","What share of trade in services do data-driven services now constitute, according to Figure 7.4?","About half of trade in services; the caption says this trade has grown exponentially since 1990.",[(258,"Since 1990, the global trade in data-driven services has grown")]),
    ("WDR-T17","test","multi_hop","Combine Figures 1.3 and 1.4: name one positive data pathway shown and one governance level at which relevant policy is organized.","One positive pathway is better policy making/service delivery; the governance framework is organized at national and international levels.",[(54,"Better policy making"),(59,"Data governance layers at the national and international levels")]),
    ("WDR-T18","test","multi_hop","Using the Croatia map discussion and Table 2.2, state why granular poverty maps matter and one low-income NSO dissemination practice value.","The map helped target EU antipoverty funds toward the poorest areas; 30% of low-income NSOs use an advance release calendar.",[(77,"Croatia responded with proposals for new geographical subdivisions that concentrate EU funds in the poorest areas."),(84,"NSO uses advance release calendar")]),
    ("WDR-T19","test","multi_hop","Combine Figures 7.4 and 7.6: how has data-driven trade changed, and what fiscal issue does the East Asia chart quantify?","Data-driven services grew exponentially and now make up half of services trade; Figure 7.6 shows indirect tax potential from B2C e-commerce in East Asia.",[(258,"grown exponentially and now constitutes half of trade in services"),(268,"Figure shows the indirect tax potential of business-to-consumer e-commerce.")]),
    ("WDR-T20","test","multi_hop","Using Tables 8.1 and 8.2, name one data-preservation action and one institution type involved in data-governance arrangements.","Catalog data systematically and include dictionaries/creation notes; a centralized governance unit may be embedded in an NSO or digital-economy ministry.",[(288,"Classify and catalog data systematically so they can be found easily."),(292,"Centralized approach: data governance agency/unit embedded in")]),
    ("WDR-T21","test","text","What kinds of records does the report give as examples of administrative data?","Examples include birth, marriage, and death records; identification, population, health, education, tax, and trade-flow records.",[(75,"also include data used by governments")]),
    ("WDR-T22","test","text","What does the report say about the coverage and format of traditional public-intent data?","They are generally designed to represent the relevant population; their formats can support inferential analysis by researchers and government officials.",[(48,"Data collected for public policy purposes are almost")]),
    ("WDR-T23","test","text","What safeguards does the report say should be in place to avoid harmful uses of data?","Robust data-protection laws, independent oversight, and legal and technological safeguards for confidentiality and against misuse.",[(79,"robust data protection laws"),(79,"oversight"),(79,"safeguard the confidentiality of individuals")]),
    ("WDR-T24","test","text","According to the report, what should a national data strategy be aligned with, and how should it be implemented?","It should align with a country's priorities and be devolved into action plans with clear targets.",[(290,"develop a national data strategy in line with the country’s priorities")]),
    # Six unanswerable test questions; search terms and absence verification are recorded below.
    ("WDR-T25","test","unanswerable","What was the 2025 market share of DeepSeek among generative-AI providers?","ABSTAIN: this 2021 report does not provide a 2025 DeepSeek market share.",[]),
    ("WDR-T26","test","unanswerable","What was the average monthly salary of a data engineer in Vietnam in 2020?","ABSTAIN: the report does not give this occupation-specific salary.",[]),
    ("WDR-T27","test","unanswerable","How many Facebook accounts were active in Vietnam in 2020?","ABSTAIN: the report does not give this country-specific account count.",[]),
    ("WDR-T28","test","unanswerable","What was the exact ransomware payment total in the world in 2020?","ABSTAIN: no such global total is reported.",[]),
    ("WDR-T29","test","unanswerable","What was the number of data scientists employed by the Vietnamese government in 2020?","ABSTAIN: the report does not give this employment count.",[]),
    ("WDR-T30","test","unanswerable","What was DeepSeek's revenue in 2025?","ABSTAIN: the report predates DeepSeek and contains no 2025 revenue evidence.",[]),
    # Dev: 15 answerable (three per evidence family) and three missing-evidence cases.
    ("WDR-D01","dev","table","In Table 1.1, give one example of new private-intent data.","Just-in-time digital data on individual behavior or choices from digital platforms.",[(48,"Just-in-time digital data")]),
    ("WDR-D02","dev","table","For high-income countries, what is Table 6.1's average score for public-intent data?","69.",[(227,"High-income                               73                       59                       43                          86                          69")]),
    ("WDR-D03","dev","table","At the 'Archive and preserve' stage in Table 8.1, what documentation should accompany preserved data?","Data dictionaries and notes on how the data were created.",[(288,"Include data dictionaries and notes on how data were created.")]),
    ("WDR-D04","dev","image_diagram","What broad stages does Figure 3.1 show data moving through on the way to development?","Data production and collection, data-driven decision-making, and development outcomes.",[(114,"The role of data in the production process: Pathways to development")]),
    ("WDR-D05","dev","image_diagram","In Figure 3.2, name one technology used by data-driven businesses.","Examples include big-data analytics, IoT/smart devices, or distributed-ledger transactions.",[(116,"Big data analytics for insights and")]),
    ("WDR-D06","dev","image_diagram","What three phases are named in Figure 9.7 for cross-border collaboration?","Establishing fundamentals, initiating data flows, and optimizing the system.",[(340,"Establishing fundamentals")]),
    ("WDR-D07","dev","chart","What does Figure 3.3 say about platform firms in some lower-income countries?","They are numerous in some lower-income countries but tend to be small.",[(117,"Platform firms are numerous in some lower-income countries but tend to be small")]),
    ("WDR-D08","dev","chart","What comparison is made in Figure 3.6 about internet traffic in low- and middle-income countries?","It is concentrated in several US-based firms.",[(119,"Internet traffic in low- and middle-income countries is concentrated in")]),
    ("WDR-D09","dev","chart","What does Figure 8.3 state about separate data-governance entities in low-income countries?","No low-income countries, and few lower-middle-income countries, have a separate entity; most embed it in another government institution.",[(293,"No low-income and few lower-middle-income countries have a separate data governance entity")]),
    ("WDR-D10","dev","multi_hop","Combine Box 2.1 and Figure 1.1: what population-data activity does a census perform, and how did the share of people counted change?","Censuses aim to enumerate the population systematically; the share counted rose from about 1 in 10 in 1850 to 9 in 10 today.",[(75,"Censuses aim to systematically enumerate"),(46,"The share of people counted in a census grew from about 1 in 10 in 1850 to")]),
    ("WDR-D11","dev","multi_hop","Combine Figure 7.4 and Table 7.2: what has grown in services trade and what kind of policy provisions does the table compare?","Data-driven services trade has grown exponentially; Table 7.2 compares digital-trade agreement provisions such as data flows and localization.",[(258,"global trade in data-driven services has grown"),(264,"Key provisions on digital trade in recent trade agreements")]),
    ("WDR-D12","dev","multi_hop","Combine Figure 8.2 with Table 8.3: name one governance function and one purpose of a data trust.","Strategic planning develops strategies; data trusts help enable trusted sharing and use of sensitive data through accountability mechanisms.",[(290,"Strategic planning"),(298,"Enable trusted sharing")]),
    ("WDR-D13","dev","text","How are machine-generated data described in Box 2.1?","They are generated automatically by sensors, applications, or computer processes without human interaction.",[(75,"automatically generated by a sensor")]),
    ("WDR-D14","dev","text","Why may private-firm data be difficult to prepare for public use?","Private firms have little incentive to curate their data for sharing.",[(48,"private firms have little")]),
    ("WDR-D15","dev","text","What does the report say only a few NSOs do to understand demand for data?","Use user-satisfaction surveys.",[(84,"only a few NSOs")]),
    ("WDR-D16","dev","unanswerable","What was DeepSeek's 2025 market share?","ABSTAIN: unsupported by this 2021 report.",[]),
    ("WDR-D17","dev","unanswerable","How many data engineers did Vietnam employ in 2020?","ABSTAIN: unsupported occupation-specific count.",[]),
    ("WDR-D18","dev","unanswerable","What was Facebook's exact 2020 user count in Vietnam?","ABSTAIN: unsupported country-specific count.",[]),
]

ABSENCE_TERMS = {
    "WDR-T25": ["DeepSeek 2025 market share"],
    "WDR-T26": ["data engineer monthly salary"],
    "WDR-T27": ["Facebook accounts in Vietnam 2020"],
    "WDR-T28": ["global ransomware payment total in 2020"],
    "WDR-T29": ["data scientists employed by Vietnamese government"],
    "WDR-T30": ["DeepSeek revenue in 2025"],
    "WDR-D16": ["DeepSeek 2025 market share"],
    "WDR-D17": ["data engineers employed in Vietnam"],
    "WDR-D18": ["Facebook exact user count in Vietnam"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def main() -> None:
    page_text = {int(row["pdf_page"]): row for row in (json.loads(x) for x in PAGES.read_text(encoding="utf-8").splitlines())}
    # Match exact anchors against the extracted page text file (not generated chunks).
    extracted = (ADV / "world_development_report_2021_layout.txt").read_text(encoding="utf-8", errors="replace").split("\f")
    rows = []
    missing_anchors = []
    for case_id, split, group, question, answer, specs in ROWS:
        gold = []
        for page_no, anchor in specs:
            text = norm(extracted[page_no - 1])
            if norm(anchor) not in text:
                missing_anchors.append(f"{case_id}, page {page_no}: {anchor}")
            gold.append({"source_id": "wdr21", "page": str(page_no), "evidence": anchor,
                         "region": f"PDF page {page_no}; find the named table/figure/map or the cited text passage"})
        rows.append({"id": case_id, "split": split, "primary_group": group, "question": question,
                     "answerable": group != "unanswerable", "expected_answer": answer,
                     "gold": gold, "absence_audit_terms": ABSENCE_TERMS.get(case_id, [])})
    if missing_anchors:
        raise ValueError("Evidence anchors not found:\n" + "\n".join(missing_anchors))
    counts = {s: {g: sum(r["split"] == s and r["primary_group"] == g for r in rows)
                  for g in sorted({r["primary_group"] for r in rows})} for s in ("dev", "test")}
    if len(rows) != 48 or sum(counts["test"].get(g, 0) for g in ("table","image_diagram","chart","multi_hop","text")) != 24:
        raise ValueError(f"Unexpected benchmark size: {counts}")
    test_answerable_groups = {g: counts["test"].get(g, 0) for g in ("table","image_diagram","chart","multi_hop","text")}
    if test_answerable_groups != {"table":6,"image_diagram":5,"chart":5,"multi_hop":4,"text":4} or counts["test"].get("unanswerable") != 6:
        raise ValueError(f"Test distribution does not meet ADVANCED.md: {counts}")
    for case_id, terms in ABSENCE_TERMS.items():
        if case_id not in {r["id"] for r in rows}:
            continue
        haystack = norm(" ".join(extracted))
        present = [term for term in terms if norm(term) in haystack]
        if present:
            raise ValueError(f"Unanswerable audit term unexpectedly occurs for {case_id}: {present}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ADV / "benchmark_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "counts": counts, "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
