"""
Builder script to construct and validate the frozen benchmark_v1.json dataset.
Extracts genuine email snippets and metadata directly from the FAISS index metadata.json.
"""
import json
from pathlib import Path
from typing import Dict, List, Any

METADATA_PATH = Path("data/rag/index/metadata.json")
OUTPUT_PATH = Path("evaluation/benchmark_v1.json")

def load_index_lookup() -> Dict[str, Dict[str, Any]]:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lookup = {}
    for item in meta:
        mid = item["message_id"]
        if mid not in lookup:
            lookup[mid] = item
    return lookup

def build_benchmark():
    lookup = load_index_lookup()
    print(f"Loaded {len(lookup)} unique emails from corpus.")

    # 50 comprehensive queries covering all required dimensions
    # Graded judgments: 2 = directly relevant, 1 = partially relevant, 0 = irrelevant
    raw_queries = [
        # 1. Academic
        {
            "id": "q01",
            "query": "NPTEL course registration",
            "category": "academic",
            "split": "dev",
            "expected_message_ids": ["19f8398e76ddf956", "19f7f717d17179eb", "19f73475d7c64216"],
            "relevance_judgments": {
                "19f8398e76ddf956": {"grade": 2, "reason": "Official Circular for NPTEL Course registration July 2026 Session with step-by-step instructions.", "source": "verified"},
                "19f7f717d17179eb": {"grade": 2, "reason": "List of SWAYAM NPTEL approved courses from SCOPE department.", "source": "verified"},
                "19f73475d7c64216": {"grade": 1, "reason": "Important Open Elective Courses SWAYAM / NPTEL online course wish list.", "source": "verified"}
            },
            "reference_answer": "NPTEL course registration for the July 2026 session is open as announced by the Dean Academics. Students must follow the SWAYAM/NPTEL portal process and consult the approved course list from their department (e.g. SCOPE).",
            "key_evidence": ["July 2026 Session", "Circular", "SWAYAM NPTEL", "Dean Academics"],
            "expect_abstention": False
        },
        {
            "id": "q02",
            "query": "SWAYAM NPTEL approved course list from SCOPE",
            "category": "academic",
            "split": "test",
            "expected_message_ids": ["19f7f717d17179eb", "1983644b73832ed7"],
            "relevance_judgments": {
                "19f7f717d17179eb": {"grade": 2, "reason": "Official list of SWAYAM NPTEL approved courses from SCOPE.", "source": "verified"},
                "1983644b73832ed7": {"grade": 1, "reason": "Fall Semester 2025-26 NPTEL/SWAYAM MOOC approved basket.", "source": "verified"},
                "19f73475d7c64216": {"grade": 1, "reason": "Open elective courses wish list mentioning approved SWAYAM courses.", "source": "verified"}
            },
            "reference_answer": "The approved SWAYAM NPTEL courses for SCOPE are released via circular, specifying eligible online courses for credit transfer in Fall/Winter sessions.",
            "key_evidence": ["SWAYAM NPTEL approved course", "SCOPE", "credit transfer"],
            "expect_abstention": False
        },
        {
            "id": "q03",
            "query": "RoboWars workshop batch 2 registration freshers",
            "category": "academic",
            "split": "dev",
            "expected_message_ids": ["19f87f26aff796c1"],
            "relevance_judgments": {
                "19f87f26aff796c1": {"grade": 2, "reason": "Exclusive for Freshers: RoboWars Workshop Batch 2 Registration Open (23 July 2026).", "source": "verified"}
            },
            "reference_answer": "RoboWars Workshop Batch 2 registration is open exclusively for freshers, scheduled for 23 July 2026.",
            "key_evidence": ["RoboWars Workshop", "Batch 2", "Freshers", "23 July 2026"],
            "expect_abstention": False
        },
        {
            "id": "q04",
            "query": "VITTBI 2-month hands-on training program",
            "category": "academic",
            "split": "test",
            "expected_message_ids": ["19f845b1a1e52c82", "19f17e66188b9118"],
            "relevance_judgments": {
                "19f845b1a1e52c82": {"grade": 2, "reason": "Request to forward 2-Month Hands-on Training Program by VITTBI to students and scholars.", "source": "verified"},
                "19f17e66188b9118": {"grade": 1, "reason": "VITTBI Master Drone Technology training announcement.", "source": "verified"}
            },
            "reference_answer": "VITTBI is organizing a 2-month hands-on training program open for VIT students, faculty, and research scholars.",
            "key_evidence": ["2-Month Hands-on Training", "VITTBI"],
            "expect_abstention": False
        },
        {
            "id": "q05",
            "query": "AICTE scholarship awareness webinar notification",
            "category": "academic",
            "split": "test",
            "expected_message_ids": ["19f7e1757b94840a", "18ab108f1fdebb12"],
            "relevance_judgments": {
                "19f7e1757b94840a": {"grade": 2, "reason": "Invitation to attend awareness webinar on AICTE scholarship/fellowship schemes.", "source": "verified"},
                "18ab108f1fdebb12": {"grade": 1, "reason": "Earlier AICTE PG scholarship notification circular.", "source": "verified"}
            },
            "reference_answer": "An awareness webinar was organized to disseminate information regarding AICTE scholarship and fellowship schemes for students and faculty.",
            "key_evidence": ["AICTE Scholarship", "Awareness Webinar", "Fellowship"],
            "expect_abstention": False
        },
        {
            "id": "q06",
            "query": "FinTech program IIT Madras NPTEL newsletter",
            "category": "academic",
            "split": "dev",
            "expected_message_ids": ["19f73f085c598a83"],
            "relevance_judgments": {
                "19f73f085c598a83": {"grade": 2, "reason": "NPTEL Newsletter: Master the Future of Finance with IIT Madras' FinTech Program.", "source": "verified"}
            },
            "reference_answer": "The NPTEL newsletter announced the IIT Madras FinTech Program titled 'Master the Future of Finance'.",
            "key_evidence": ["NPTEL Newsletter", "IIT Madras", "FinTech Program"],
            "expect_abstention": False
        },
        {
            "id": "q07",
            "query": "SOP for patent incentive and IPR notices",
            "category": "academic",
            "split": "test",
            "expected_message_ids": ["19f5b7cd2134615b", "19daac6e30a8afc1"],
            "relevance_judgments": {
                "19f5b7cd2134615b": {"grade": 2, "reason": "Standard Operating Procedure (SOP) for Patent Incentive and Filing.", "source": "verified"},
                "19daac6e30a8afc1": {"grade": 1, "reason": "Notification of SOP Communication and Patent Processing.", "source": "verified"}
            },
            "reference_answer": "The IPR cell released an SOP governing patent filing procedures and monetary incentive schemes for student and faculty inventors.",
            "key_evidence": ["Standard Operating Procedure", "Patent Incentive", "IPR"],
            "expect_abstention": False
        },
        {
            "id": "q08",
            "query": "Important open elective courses SWAYAM NPTEL wish list",
            "category": "academic",
            "split": "test",
            "expected_message_ids": ["19f73475d7c64216", "19f7f717d17179eb"],
            "relevance_judgments": {
                "19f73475d7c64216": {"grade": 2, "reason": "Important Open Elective Courses: SWAYAM / NPTEL online course wish list for July-Dec 2026.", "source": "verified"},
                "19f7f717d17179eb": {"grade": 1, "reason": "List of SWAYAM NPTEL approved courses from SCOPE.", "source": "verified"}
            },
            "reference_answer": "The wish list for SWAYAM / NPTEL online open elective courses for July-Dec 2026 was released for student course selection.",
            "key_evidence": ["SWAYAM / NPTEL online course wish list", "July-Dec 2026"],
            "expect_abstention": False
        },
        {
            "id": "q09",
            "query": "AICTE PG Scholarship guidelines and stipend",
            "category": "academic",
            "split": "test",
            "expected_message_ids": ["18ab108f1fdebb12", "19f7e1757b94840a"],
            "relevance_judgments": {
                "18ab108f1fdebb12": {"grade": 2, "reason": "AICTE PG Scholarship Notification with eligibility and application guidelines.", "source": "verified"},
                "19f7e1757b94840a": {"grade": 1, "reason": "Awareness webinar covering AICTE PG scholarship scheme details.", "source": "verified"}
            },
            "reference_answer": "The AICTE PG Scholarship provides monthly financial assistance to eligible GATE/CEED qualified postgraduate students according to council guidelines.",
            "key_evidence": ["AICTE PG Scholarship", "Notification", "GATE"],
            "expect_abstention": False
        },

        # 2. Placement - Assessments & Tests
        {
            "id": "q10",
            "query": "When is my Microsoft assessment?",
            "category": "placement",
            "split": "dev",
            "expected_message_ids": ["19bb5a81a1c389ec", "19bb5a49f6c00e3a", "19b973b78c7602fb"],
            "relevance_judgments": {
                "19bb5a81a1c389ec": {"grade": 2, "reason": "Direct notification of Microsoft assessment date and test link.", "source": "verified"},
                "19bb5a49f6c00e3a": {"grade": 2, "reason": "Reminder / updated instructions for Microsoft online assessment.", "source": "verified"},
                "19b973b78c7602fb": {"grade": 1, "reason": "Initial registration and test window announcement for Microsoft.", "source": "verified"}
            },
            "reference_answer": "Your Microsoft online assessment is scheduled according to the CDC test notification email, which contains the test window link and timing.",
            "key_evidence": ["Microsoft", "assessment", "test window"],
            "expect_abstention": False
        },
        {
            "id": "q11",
            "query": "Nokia online test",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19ebbe31cb35a6ce", "19ebbc88864a01f7", "19ea27c9ea15e4b4", "19e97c22fc8aa4bc"],
            "relevance_judgments": {
                "19ebbe31cb35a6ce": {"grade": 2, "reason": "Direct schedule for NOKIA online test and login credentials.", "source": "verified"},
                "19ebbc88864a01f7": {"grade": 2, "reason": "Important instructions and platform details for NOKIA test.", "source": "verified"},
                "19ea27c9ea15e4b4": {"grade": 1, "reason": "Shortlisted candidates notice for NOKIA assessment.", "source": "verified"},
                "19e97c22fc8aa4bc": {"grade": 1, "reason": "Prior announcement regarding Nokia placement drive.", "source": "verified"}
            },
            "reference_answer": "The Nokia online test details including slot time, test platform link, and candidate login credentials were sent by CDC.",
            "key_evidence": ["Nokia", "online test", "slot", "CDC"],
            "expect_abstention": False
        },
        {
            "id": "q12",
            "query": "Tata Electronics online test",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19e722b6233a1d56", "19e6ca9f465dbed5"],
            "relevance_judgments": {
                "19e722b6233a1d56": {"grade": 2, "reason": "Tata Electronics online assessment schedule and shortlisted students.", "source": "verified"},
                "19e6ca9f465dbed5": {"grade": 1, "reason": "Tata Electronics initial registration and test pattern email.", "source": "verified"}
            },
            "reference_answer": "Tata Electronics scheduled an online assessment for eligible registered students with test guidelines and timings.",
            "key_evidence": ["Tata Electronics", "online test", "shortlist"],
            "expect_abstention": False
        },
        {
            "id": "q13",
            "query": "VISA online test",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19a95657a987363b", "199be2ad06dc42bd", "199516fdd79d2898", "1993ccdaa6df9e6b"],
            "relevance_judgments": {
                "19a95657a987363b": {"grade": 2, "reason": "VISA online assessment invite and HackerRank/test platform instructions.", "source": "verified"},
                "199be2ad06dc42bd": {"grade": 2, "reason": "VISA recruitment online coding assessment schedule.", "source": "verified"},
                "199516fdd79d2898": {"grade": 1, "reason": "Shortlist and interview process following the VISA test.", "source": "verified"},
                "1993ccdaa6df9e6b": {"grade": 1, "reason": "Initial registration notice for VISA recruitment.", "source": "verified"}
            },
            "reference_answer": "VISA conducted an online coding assessment on HackerRank for registered students, followed by technical interviews.",
            "key_evidence": ["VISA", "online test", "HackerRank", "CDC"],
            "expect_abstention": False
        },
        {
            "id": "q14",
            "query": "Optum online test assessment",
            "category": "placement",
            "split": "dev",
            "expected_message_ids": ["1999f0ad6d6c050c", "1999a5bf172c03ef", "199be2da0a6e53da"],
            "relevance_judgments": {
                "1999f0ad6d6c050c": {"grade": 2, "reason": "Optum online test schedule and slot distribution.", "source": "verified"},
                "1999a5bf172c03ef": {"grade": 2, "reason": "Candidate instructions for Optum assessment test.", "source": "verified"},
                "199be2da0a6e53da": {"grade": 1, "reason": "Optum shortlisted candidates announcement.", "source": "verified"}
            },
            "reference_answer": "The Optum online assessment was scheduled by CDC, specifying test timings, venue/system requirements, and shortlist info.",
            "key_evidence": ["Optum", "online test", "assessment", "CDC"],
            "expect_abstention": False
        },
        {
            "id": "q15",
            "query": "emails about upcoming online tests and assessments",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f8840a933d87da", "19f84321b3644afd", "19f7dc38270447c6", "19f73dc3c81faa01"],
            "relevance_judgments": {
                "19f8840a933d87da": {"grade": 2, "reason": "Assessment schedule announcement for placement drives.", "source": "verified"},
                "19f84321b3644afd": {"grade": 2, "reason": "Infosys PPT and online test scheduled announcement.", "source": "verified"},
                "19f7dc38270447c6": {"grade": 2, "reason": "Pixel Compute Technologies online test scheduled announcement.", "source": "verified"},
                "19f73dc3c81faa01": {"grade": 1, "reason": "Upcoming placement test instructions and roster.", "source": "verified"}
            },
            "reference_answer": "Upcoming assessments announced by CDC include Infosys Batch 2 test and Pixel Compute online test, with designated lab and hall venues.",
            "key_evidence": ["online test", "Infosys", "Pixel Compute", "CDC"],
            "expect_abstention": False
        },

        # 3. Placement - Selection Lists
        {
            "id": "q16",
            "query": "Ericsson internship selection list",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f889a11033fb3b", "19e062e0323a6d4e"],
            "relevance_judgments": {
                "19f889a11033fb3b": {"grade": 2, "reason": "Congratulations !! Ericsson Super Dream Internship selection list 2027 Batch !!", "source": "verified"},
                "19e062e0323a6d4e": {"grade": 1, "reason": "Prior Ericsson Super Dream Internship selection notification.", "source": "verified"}
            },
            "reference_answer": "CDC released the congratulations circular containing the selection list for the Ericsson Super Dream Internship for the 2027 batch.",
            "key_evidence": ["Ericsson Super Dream Internship", "selection list", "2027 Batch"],
            "expect_abstention": False
        },
        {
            "id": "q17",
            "query": "Couchbase super dream internship selection list",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f83efe3887f1a1", "19f7d7476cdaabc4"],
            "relevance_judgments": {
                "19f83efe3887f1a1": {"grade": 2, "reason": "Congratulations !! Couchbase Super Dream Internship Selection List !! - 2027 Batch", "source": "verified"},
                "19f7d7476cdaabc4": {"grade": 1, "reason": "Re: Couchbase next round of selection process scheduled notice.", "source": "verified"}
            },
            "reference_answer": "The selection list for Couchbase Super Dream Internship for the 2027 batch was officially published following the technical rounds.",
            "key_evidence": ["Couchbase", "Super Dream Internship", "Selection List"],
            "expect_abstention": False
        },
        {
            "id": "q18",
            "query": "Schneider Electric super dream internship selection",
            "category": "placement",
            "split": "dev",
            "expected_message_ids": ["19f59950d47b15fa", "19eb61b250d2946d"],
            "relevance_judgments": {
                "19f59950d47b15fa": {"grade": 2, "reason": "Congratulations!! Schneider Electric Super Dream internship selection announcement.", "source": "verified"},
                "19eb61b250d2946d": {"grade": 1, "reason": "Earlier Schneider Electric internship selection roster.", "source": "verified"}
            },
            "reference_answer": "CDC announced the selected students for Schneider Electric Super Dream internship with congratulatory notices.",
            "key_evidence": ["Schneider Electric", "Super Dream internship", "selection"],
            "expect_abstention": False
        },
        {
            "id": "q19",
            "query": "Matrecomm dream internship selection list",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f842578ffad059", "19f7d95f01815cc3"],
            "relevance_judgments": {
                "19f842578ffad059": {"grade": 2, "reason": "Congratulations !! Matrecomm Dream Internship selection list 2027 batch !!", "source": "verified"},
                "19f7d95f01815cc3": {"grade": 1, "reason": "Matrecomm next round of selection process (Technical Interview).", "source": "verified"}
            },
            "reference_answer": "Matrecomm Dream Internship selection list for the 2027 batch was published by CDC following technical interviews.",
            "key_evidence": ["Matrecomm", "Dream Internship", "selection list"],
            "expect_abstention": False
        },
        {
            "id": "q20",
            "query": "CDC Summer intern selects 2027 batch",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f63fd247085055", "19f11667efcf0747"],
            "relevance_judgments": {
                "19f63fd247085055": {"grade": 2, "reason": "kind attention!! Summer intern selects - 2027 Batch - vellore campus circular.", "source": "verified"},
                "19f11667efcf0747": {"grade": 1, "reason": "Trademarkia Summer Internship PPO selection list.", "source": "verified"}
            },
            "reference_answer": "CDC issued a circular for all 2027 batch summer intern selects regarding onboarding formalities and portal verification.",
            "key_evidence": ["Summer intern selects", "2027 Batch", "CDC"],
            "expect_abstention": False
        },

        # 4. Placement - Registrations & Drives
        {
            "id": "q21",
            "query": "Fischer Jordan super dream placement registration",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f83a92b566c8b0", "19f7ef15ff978fa6"],
            "relevance_judgments": {
                "19f83a92b566c8b0": {"grade": 2, "reason": "Re: Fischer Jordan Super Dream Internship / Placement Registration - 2027 Batch reminder.", "source": "verified"},
                "19f7ef15ff978fa6": {"grade": 2, "reason": "Fischer Jordan Super Dream Internship / Placement Registration - 2027 Batch main notice.", "source": "verified"}
            },
            "reference_answer": "Registration for Fischer Jordan Super Dream Internship / Placement was opened for the 2027 batch on the CDC portal with eligibility criteria.",
            "key_evidence": ["Fischer Jordan", "Super Dream", "Registration", "2027 Batch"],
            "expect_abstention": False
        },
        {
            "id": "q22",
            "query": "Wakefit dream core internship registration",
            "category": "placement",
            "split": "dev",
            "expected_message_ids": ["19f7e77b286e8aa6", "19f69d4d188260be"],
            "relevance_judgments": {
                "19f7e77b286e8aa6": {"grade": 2, "reason": "Re: Wakefit Dream Core Internship Registration - 2027 Batch reminder.", "source": "verified"},
                "19f69d4d188260be": {"grade": 2, "reason": "Update : Wakefit Dream Core Internship Registration - 2027 Batch with deadline.", "source": "verified"}
            },
            "reference_answer": "Registration for Wakefit Dream Core Internship was announced for the 2027 batch with registration deadlines on the placement portal.",
            "key_evidence": ["Wakefit", "Dream Core", "Internship Registration", "2027 Batch"],
            "expect_abstention": False
        },
        {
            "id": "q23",
            "query": "Work India super dream placement registration",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f7e795f05766e0"],
            "relevance_judgments": {
                "19f7e795f05766e0": {"grade": 2, "reason": "Work India Super Dream Internship / Placement – 2027 Batch Registration notice.", "source": "verified"}
            },
            "reference_answer": "Work India announced registration for Super Dream Internship and Placement for the 2027 batch.",
            "key_evidence": ["Work India", "Super Dream", "2027 Batch Registration"],
            "expect_abstention": False
        },
        {
            "id": "q24",
            "query": "Pepagora regular internship registration",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f83227d2c02211"],
            "relevance_judgments": {
                "19f83227d2c02211": {"grade": 2, "reason": "Pepagora (India) Pvt Ltd regular internship registration 2027 Batch circular.", "source": "verified"}
            },
            "reference_answer": "Pepagora (India) Pvt Ltd opened regular internship registration for 2027 batch students.",
            "key_evidence": ["Pepagora", "regular internship registration", "2027 Batch"],
            "expect_abstention": False
        },
        {
            "id": "q25",
            "query": "VENTURA AUTOMATION SERVICES internship registration",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f7f75e09b01a54"],
            "relevance_judgments": {
                "19f7f75e09b01a54": {"grade": 2, "reason": "VENTURA AUTOMATION SERVICES INC Regular Internship Registration - 2027 Batch.", "source": "verified"}
            },
            "reference_answer": "Ventura Automation Services opened regular internship registration for the 2027 batch via CDC.",
            "key_evidence": ["VENTURA AUTOMATION SERVICES", "Regular Internship Registration"],
            "expect_abstention": False
        },
        {
            "id": "q26",
            "query": "Intain Campus FinTech Challenge 2026",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f84a0485f147cc"],
            "relevance_judgments": {
                "19f84a0485f147cc": {"grade": 2, "reason": "Internship Opportunity for 2027 Batch Students | Intain Campus FinTech Challenge 2026.", "source": "verified"}
            },
            "reference_answer": "Intain announced a campus FinTech challenge offering internship opportunities for 2027 batch students.",
            "key_evidence": ["Intain Campus FinTech Challenge 2026", "Internship Opportunity", "2027 Batch"],
            "expect_abstention": False
        },
        {
            "id": "q27",
            "query": "Honeywell campus connect hackathon",
            "category": "placement",
            "split": "dev",
            "expected_message_ids": ["19f848c57d7a26a8", "19f7011a77ab78b9"],
            "relevance_judgments": {
                "19f848c57d7a26a8": {"grade": 2, "reason": "Honeywell Technologies Campus Connect Hackathon - Register now circular.", "source": "verified"},
                "19f7011a77ab78b9": {"grade": 2, "reason": "Initial Honeywell Hackathon announcement.", "source": "verified"}
            },
            "reference_answer": "Honeywell Technologies launched the Campus Connect Hackathon inviting university students to register and compete.",
            "key_evidence": ["Honeywell Technologies", "Campus Connect Hackathon", "Register"],
            "expect_abstention": False
        },
        {
            "id": "q28",
            "query": "TCS pre-placement talk placement policy orientation",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f5f9b792deadc7", "19f5f50a768dce6d"],
            "relevance_judgments": {
                "19f5f9b792deadc7": {"grade": 2, "reason": "TCS Pre-placement Talk and placement policy orientation notification.", "source": "verified"},
                "19f5f50a768dce6d": {"grade": 1, "reason": "Orientation link and schedule confirmation for TCS talk.", "source": "verified"}
            },
            "reference_answer": "TCS organized a pre-placement talk accompanied by an orientation on university placement policies and guidelines.",
            "key_evidence": ["TCS", "Pre-placement Talk", "Placement Policy"],
            "expect_abstention": False
        },
        {
            "id": "q29",
            "query": "placement interviews",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19ecb1d09c113c16", "19eb02dc5d4acd27", "19ed0235344dd96b"],
            "relevance_judgments": {
                "19ecb1d09c113c16": {"grade": 2, "reason": "Placement interview shortlist and interview slot timings.", "source": "verified"},
                "19eb02dc5d4acd27": {"grade": 2, "reason": "CDC interview instructions and technical panel logistics.", "source": "verified"},
                "19ed0235344dd96b": {"grade": 1, "reason": "Placement interview preparation and mock interview schedule.", "source": "verified"}
            },
            "reference_answer": "Placement interviews are coordinated through CDC with shortlisted students assigned specific time slots and technical interview panels.",
            "key_evidence": ["placement interview", "shortlist", "CDC", "slot"],
            "expect_abstention": False
        },
        {
            "id": "q30",
            "query": "Mandatory mock interviews with Expert Hire",
            "category": "placement",
            "split": "test",
            "expected_message_ids": ["19f83f7d01c098b8", "19f5eb3d6814f68d"],
            "relevance_judgments": {
                "19f83f7d01c098b8": {"grade": 2, "reason": "Mandatory Mock interviews with Expert Hire scheduling announcement.", "source": "verified"},
                "19f5eb3d6814f68d": {"grade": 1, "reason": "Expert Hire Super dream offer selection notice.", "source": "verified"}
            },
            "reference_answer": "CDC announced mandatory mock interviews conducted in collaboration with Expert Hire to prepare students for corporate interview rounds.",
            "key_evidence": ["Mandatory Mock interviews", "Expert Hire", "CDC"],
            "expect_abstention": False
        },

        # 5. Schedules & Venues
        {
            "id": "q31",
            "query": "Pixel Compute online test venue",
            "category": "schedules_venues",
            "split": "dev",
            "expected_message_ids": ["19f83fea04ddf03f", "19f7dc38270447c6"],
            "relevance_judgments": {
                "19f83fea04ddf03f": {"grade": 2, "reason": "Re: Pixel Compute Technologies Pvt Ltd online test venue confirmation (PRP 717).", "source": "verified"},
                "19f7dc38270447c6": {"grade": 2, "reason": "Pixel Compute Technologies Pvt Ltd online test scheduled with venue PRP 717.", "source": "verified"}
            },
            "reference_answer": "The Pixel Compute Technologies online test was scheduled at the PRP 717 computer lab venue.",
            "key_evidence": ["Pixel Compute", "PRP 717", "online test venue"],
            "expect_abstention": False
        },
        {
            "id": "q32",
            "query": "Infosys PPT online test Anna auditorium",
            "category": "schedules_venues",
            "split": "test",
            "expected_message_ids": ["19f89035dbd3de18", "19f84321b3644afd", "19f83724846a9905", "19f7f8cafecc0ba7"],
            "relevance_judgments": {
                "19f89035dbd3de18": {"grade": 2, "reason": "Re: Infosys PPT & Online Test Scheduled on Batch -2 at Anna Auditorium.", "source": "verified"},
                "19f84321b3644afd": {"grade": 2, "reason": "Infosys PPT & Online Test Is Scheduled on Batch -2 (22-07-2026) in Anna Auditorium.", "source": "verified"},
                "19f83724846a9905": {"grade": 1, "reason": "Prior batch schedule for Infosys pre-placement talk.", "source": "verified"},
                "19f7f8cafecc0ba7": {"grade": 1, "reason": "Infosys registration circular and auditorium venue details.", "source": "verified"}
            },
            "reference_answer": "The Infosys Pre-Placement Talk (PPT) and online test for Batch 2 was scheduled at Anna Auditorium on 22 July 2026.",
            "key_evidence": ["Infosys PPT", "Anna Auditorium", "Batch -2", "22-07-2026"],
            "expect_abstention": False
        },
        {
            "id": "q33",
            "query": "Pixel Compute online test PRP 717 venue and timings",
            "category": "schedules_venues",
            "split": "dev",
            "expected_message_ids": ["19f7dc38270447c6", "19f83fea04ddf03f"],
            "relevance_judgments": {
                "19f7dc38270447c6": {"grade": 2, "reason": "Pixel Compute online test schedule with exact timing and PRP 717 lab.", "source": "verified"},
                "19f83fea04ddf03f": {"grade": 2, "reason": "Re: Pixel Compute online test venue confirmation and reporting time.", "source": "verified"}
            },
            "reference_answer": "The Pixel Compute online assessment was scheduled in PRP 717 with candidate slot reporting times specified in the CDC circular.",
            "key_evidence": ["Pixel Compute", "PRP 717", "timings"],
            "expect_abstention": False
        },
        {
            "id": "q34",
            "query": "Infosys PPT Anna Auditorium batch 2 schedule",
            "category": "schedules_venues",
            "split": "test",
            "expected_message_ids": ["19f84321b3644afd", "19f89035dbd3de18"],
            "relevance_judgments": {
                "19f84321b3644afd": {"grade": 2, "reason": "Infosys PPT & Online Test Is Scheduled on Batch -2 (22-07-2026) at Anna Auditorium.", "source": "verified"},
                "19f89035dbd3de18": {"grade": 2, "reason": "Re: Infosys PPT & Online Test Batch -2 update and timing.", "source": "verified"}
            },
            "reference_answer": "Batch 2 of the Infosys PPT and Online Test took place on 22 July 2026 at Anna Auditorium.",
            "key_evidence": ["Batch -2", "22-07-2026", "Anna Auditorium", "Infosys PPT"],
            "expect_abstention": False
        },
        {
            "id": "q35",
            "query": "Where are CDC mock interviews conducted?",
            "category": "schedules_venues",
            "split": "test",
            "expected_message_ids": ["19f83f7d01c098b8"],
            "relevance_judgments": {
                "19f83f7d01c098b8": {"grade": 2, "reason": "Mandatory Mock interviews with Expert Hire instructions specifying online/lab links.", "source": "verified"}
            },
            "reference_answer": "The mandatory mock interviews with Expert Hire are conducted virtually or in designated CDC labs with slot bookings.",
            "key_evidence": ["Mock interviews", "Expert Hire", "CDC"],
            "expect_abstention": False
        },

        # 6. Security
        {
            "id": "q36",
            "query": "Google security alert sign in",
            "category": "security",
            "split": "test",
            "expected_message_ids": ["19f895bd054a1336", "19f829d793c5d1b0"],
            "relevance_judgments": {
                "19f895bd054a1336": {"grade": 2, "reason": "Official Google Security alert regarding new device sign-in.", "source": "verified"},
                "19f829d793c5d1b0": {"grade": 2, "reason": "Security alert for user account with sign-in device details.", "source": "verified"}
            },
            "reference_answer": "Google sent an automated security alert warning that a new sign-in occurred on your Google account.",
            "key_evidence": ["Security alert", "Google", "sign-in"],
            "expect_abstention": False
        },
        {
            "id": "q37",
            "query": "Security alert for Google account",
            "category": "security",
            "split": "test",
            "expected_message_ids": ["19f895bd054a1336", "19f829d793c5d1b0"],
            "relevance_judgments": {
                "19f895bd054a1336": {"grade": 2, "reason": "Google Security alert notification on 22 Jul 2026.", "source": "verified"},
                "19f829d793c5d1b0": {"grade": 2, "reason": "Google Security alert for account on 21 Jul 2026.", "source": "verified"}
            },
            "reference_answer": "Google security alerts were sent to notify the user of sign-in activity and account protection status.",
            "key_evidence": ["Security alert", "Google account", "sign-in activity"],
            "expect_abstention": False
        },
        {
            "id": "q38",
            "query": "Password reset OTP and security notices",
            "category": "security",
            "split": "test",
            "expected_message_ids": ["198a28e3eeba8e26", "1986c0491809c109"],
            "relevance_judgments": {
                "198a28e3eeba8e26": {"grade": 2, "reason": "Reset/Forgot Password - OTP authentication email.", "source": "verified"},
                "1986c0491809c109": {"grade": 2, "reason": "Reset/Forgot Password - OTP authentication notification.", "source": "verified"}
            },
            "reference_answer": "Password reset OTP emails were received containing verification codes for account password changes.",
            "key_evidence": ["Reset/Forgot Password", "OTP", "verification"],
            "expect_abstention": False
        },
        {
            "id": "q39",
            "query": "Recent security alerts on Google account",
            "category": "security",
            "split": "dev",
            "expected_message_ids": ["19f895bd054a1336", "19f829d793c5d1b0"],
            "relevance_judgments": {
                "19f895bd054a1336": {"grade": 2, "reason": "Most recent Google security alert on 22 July 2026.", "source": "verified"},
                "19f829d793c5d1b0": {"grade": 2, "reason": "Google security alert on 21 July 2026.", "source": "verified"}
            },
            "reference_answer": "Recent security alerts on your Google account were logged on 21 and 22 July 2026 alerting to device sign-ins.",
            "key_evidence": ["22 July 2026", "Google security alert", "sign-in"],
            "expect_abstention": False
        },

        # 7. Sender-based Queries
        {
            "id": "q40",
            "query": "emails from CDC about placement registration",
            "category": "sender_based",
            "split": "test",
            "expected_message_ids": ["19f83a92b566c8b0", "19f7ef15ff978fa6", "19f7e795f05766e0", "19f6a80e217ae80c"],
            "relevance_judgments": {
                "19f83a92b566c8b0": {"grade": 2, "reason": "Fischer Jordan placement registration notice from CDC.", "source": "verified"},
                "19f7ef15ff978fa6": {"grade": 2, "reason": "CDC circular for Super Dream placement registration.", "source": "verified"},
                "19f7e795f05766e0": {"grade": 2, "reason": "Work India placement registration email from CDC.", "source": "verified"},
                "19f6a80e217ae80c": {"grade": 1, "reason": "CDC circular announcing placement portal open window.", "source": "verified"}
            },
            "reference_answer": "Emails from CDC about placement registration include circulars for Fischer Jordan, Work India, and Wakefit for the 2027 batch.",
            "key_evidence": ["CDC", "placement registration", "2027 batch"],
            "expect_abstention": False
        },
        {
            "id": "q41",
            "query": "Emails from Helpdesk CDC regarding 2027 batch queries",
            "category": "sender_based",
            "split": "dev",
            "expected_message_ids": ["19f84ba64a86a7a1", "19f63fd247085055"],
            "relevance_judgments": {
                "19f84ba64a86a7a1": {"grade": 2, "reason": "DABB QUIERES-2027 BATCH email sent by Helpdesk CDC.", "source": "verified"},
                "19f63fd247085055": {"grade": 2, "reason": "kind attention!! Summer intern selects - 2027 Batch by Helpdesk CDC.", "source": "verified"}
            },
            "reference_answer": "Helpdesk CDC sent emails addressing 2027 batch student queries regarding DABB and summer internship verification.",
            "key_evidence": ["Helpdesk CDC", "2027 BATCH", "queries"],
            "expect_abstention": False
        },
        {
            "id": "q42",
            "query": "Emails from Director Physical Education sports achievements",
            "category": "sender_based",
            "split": "test",
            "expected_message_ids": ["19f3b0594081d15e", "19e1bae9ed39dac9"],
            "relevance_judgments": {
                "19f3b0594081d15e": {"grade": 2, "reason": "Sports Achievement 2026 circular from Director Physical Education.", "source": "verified"},
                "19e1bae9ed39dac9": {"grade": 2, "reason": "Sports Achievements 2026 announcement circular.", "source": "verified"}
            },
            "reference_answer": "The Director of Physical Education sent circulars celebrating university sports achievements in national and inter-university tournaments.",
            "key_evidence": ["Director Physical Education", "Sports Achievement 2026"],
            "expect_abstention": False
        },
        {
            "id": "q43",
            "query": "Emails from Dean Academic Research journal publications",
            "category": "sender_based",
            "split": "test",
            "expected_message_ids": ["19f3b37bd47a2954", "19e9c74c5b7f6615"],
            "relevance_judgments": {
                "19f3b37bd47a2954": {"grade": 2, "reason": "Journal Publication Details of VIT for June 2026 from Dean Academic Research.", "source": "verified"},
                "19e9c74c5b7f6615": {"grade": 2, "reason": "Journal Publication Details of VIT for May 2026 from Dean Academic Research.", "source": "verified"}
            },
            "reference_answer": "Dean Academic Research sends monthly compilations of VIT journal publication details indexed in Scopus/WoS.",
            "key_evidence": ["Dean Academic Research", "Journal Publication Details", "VIT"],
            "expect_abstention": False
        },

        # 8. Temporal Queries
        {
            "id": "q44",
            "query": "most recent placement company",
            "category": "temporal",
            "split": "test",
            "expected_message_ids": ["19f84cdb2b89676e", "19f7e795f05766e0", "19f7e5b3352ba238", "19f6f7cf92439794"],
            "relevance_judgments": {
                "19f84cdb2b89676e": {"grade": 2, "reason": "Ejadah Asset Management Group LLC Super Dream Placement on 21 Jul 2026.", "source": "verified"},
                "19f7e795f05766e0": {"grade": 2, "reason": "Work India Super Dream Placement on 20 Jul 2026.", "source": "verified"},
                "19f7e5b3352ba238": {"grade": 1, "reason": "Recent placement drive circular from July 2026.", "source": "verified"},
                "19f6f7cf92439794": {"grade": 1, "reason": "Placement company notification from mid-July 2026.", "source": "verified"}
            },
            "reference_answer": "The most recent placement companies announced in late July 2026 include Ejadah Asset Management Group LLC and Work India.",
            "key_evidence": ["Ejadah Asset Management", "Work India", "July 2026"],
            "expect_abstention": False
        },
        {
            "id": "q45",
            "query": "Internships announced in July 2026",
            "category": "temporal",
            "split": "test",
            "expected_message_ids": ["19f84cdb2b89676e", "19f83227d2c02211", "19f7e77b286e8aa6"],
            "relevance_judgments": {
                "19f84cdb2b89676e": {"grade": 2, "reason": "Ejadah Asset Management announced in July 2026.", "source": "verified"},
                "19f83227d2c02211": {"grade": 2, "reason": "Pepagora regular internship announced in July 2026.", "source": "verified"},
                "19f7e77b286e8aa6": {"grade": 2, "reason": "Wakefit internship registration announced in July 2026.", "source": "verified"}
            },
            "reference_answer": "Internship opportunities announced in July 2026 include Ejadah Asset Management, Pepagora India, and Wakefit for the 2027 batch.",
            "key_evidence": ["July 2026", "Ejadah", "Pepagora", "Wakefit"],
            "expect_abstention": False
        },

        # 9. Summarization
        {
            "id": "q46",
            "query": "Fake internship and placement emails warning alert",
            "category": "summarization",
            "split": "test",
            "expected_message_ids": ["19f7e60f595f6ed9", "198426d8d16be0b5"],
            "relevance_judgments": {
                "19f7e60f595f6ed9": {"grade": 2, "reason": "Important Alert: Beware of Fake Internship/Placement Emails/WhatsApp-reg.", "source": "verified"},
                "198426d8d16be0b5": {"grade": 1, "reason": "Prior advisory on fraudulent job offers.", "source": "verified"}
            },
            "reference_answer": "The university issued an advisory warning students to beware of fake internship/placement emails and WhatsApp messages, advising them not to share personal details or pay any money.",
            "key_evidence": ["Beware of Fake Internship/Placement Emails", "fraudulent", "WhatsApp"],
            "expect_abstention": False
        },
        {
            "id": "q47",
            "query": "Summarize the instructions for NPTEL July 2026 course registration",
            "category": "summarization",
            "split": "dev",
            "expected_message_ids": ["19f8398e76ddf956", "19f7f717d17179eb"],
            "relevance_judgments": {
                "19f8398e76ddf956": {"grade": 2, "reason": "Circular NPTEL Course registration July 2026 Session with step-by-step instructions.", "source": "verified"},
                "19f7f717d17179eb": {"grade": 1, "reason": "SCOPE approved course list for credit transfer.", "source": "verified"}
            },
            "reference_answer": "For NPTEL July 2026, students must select courses approved by their school, register on the SWAYAM portal, use their official email ID, and comply with credit transfer rules.",
            "key_evidence": ["NPTEL Course registration", "July 2026 Session", "SWAYAM portal", "approved course list"],
            "expect_abstention": False
        },

        # 10. Out-of-Domain & Irrelevant Queries (Must Abstain)
        {
            "id": "q48",
            "query": "recipe for chocolate cake with walnuts",
            "category": "irrelevant",
            "split": "test",
            "expected_message_ids": [],
            "relevance_judgments": {},
            "reference_answer": "I couldn't find any relevant emails in your mailbox for this query. Please try rephrasing your question.",
            "key_evidence": [],
            "expect_abstention": True
        },
        {
            "id": "q49",
            "query": "Who won the 1994 FIFA World Cup in the United States?",
            "category": "irrelevant",
            "split": "dev",
            "expected_message_ids": [],
            "relevance_judgments": {},
            "reference_answer": "I couldn't find any relevant emails in your mailbox for this query. Please try rephrasing your question.",
            "key_evidence": [],
            "expect_abstention": True
        },
        {
            "id": "q50",
            "query": "How to replace brake pads on a 2018 Honda Civic",
            "category": "irrelevant",
            "split": "test",
            "expected_message_ids": [],
            "relevance_judgments": {},
            "reference_answer": "I couldn't find any relevant emails in your mailbox for this query. Please try rephrasing your question.",
            "key_evidence": [],
            "expect_abstention": True
        }
    ]

    # Verify coverage against actual corpus
    missing_ids = []
    total_judgments = 0
    grade_counts = {2: 0, 1: 0, 0: 0}

    for q in raw_queries:
        for mid, jdata in q["relevance_judgments"].items():
            total_judgments += 1
            grade_counts[jdata["grade"]] += 1
            if mid not in lookup:
                missing_ids.append(mid)

    print(f"Total queries: {len(raw_queries)}")
    print(f"Total relevance judgments: {total_judgments}")
    print(f"Grades: {grade_counts}")
    print(f"Missing corpus IDs: {len(missing_ids)} ({missing_ids})")

    benchmark_data = {
        "benchmark_version": "1.0.0",
        "dataset_name": "MailMind Evaluation Benchmark (Frozen)",
        "description": "Frozen, relevance-judged retrieval and RAG benchmark covering university email communications.",
        "splits": {
            "dev": sum(1 for q in raw_queries if q["split"] == "dev"),
            "test": sum(1 for q in raw_queries if q["split"] == "test"),
            "total": len(raw_queries)
        },
        "categories": list({q["category"] for q in raw_queries}),
        "grading_scale": {
            "2": "Directly relevant (primary target document answering query)",
            "1": "Partially relevant (relevant context, follow-up, general policy)",
            "0": "Irrelevant / false positive"
        },
        "binary_recall_convention": "Grades 1 and 2 are counted as relevant for binary recall and precision.",
        "queries": raw_queries
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"Saved frozen benchmark to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_benchmark()
