"""Synthetic PERM disclosure data generator.

Produces a dataset that mimics the column names, dtypes, value vocabularies,
missingness, and quirks of the U.S. DOL OFLC **PERM disclosure files**
(ETA Form 9089), so the rest of the pipeline — cleaning, feature engineering,
temporal validation, modelling, SHAP, fairness — can be built and tested before
any real government spreadsheet is downloaded.

    python -m src.data_prep.make_synthetic

===============================================================================
THIS IS NOT REAL DATA
===============================================================================
Every row is invented. The correlations built into this generator are *design
decisions*, not empirical findings. They exist so that a model trained on this
file learns *something* and the pipeline can be smoke-tested end to end.

No accuracy score, SHAP plot, or fairness metric computed from this file is a
result. Anything derived from it must be labelled synthetic. Real numbers come
only from the DOL files described in ``data/README.md``.
===============================================================================

Realism the generator deliberately reproduces
---------------------------------------------
1. **Rows are grouped by DECISION fiscal year, not filing year.** A real FY2023
   disclosure file contains cases *decided* in FY2023, many of which were filed
   in FY2021 or FY2022. ``RECEIVED_DATE`` therefore routinely falls outside the
   row's ``FISCAL_YEAR``. This matters: it is why a temporal split must be made
   on decision date, and why "cases filed in year X" is not what the file holds.

2. **SOC code drift.** FY2018-FY2019 rows carry 2010 SOC codes/titles
   (e.g. ``15-1132 Software Developers, Applications``); FY2020+ rows carry
   2018 SOC codes (``15-1252 Software Developers``). Any model that treats the
   raw code as a categorical level without harmonising will see the same job as
   two unrelated categories split exactly on the train/test boundary.

3. **Backlog regimes by fiscal year.** Median processing time moves a lot across
   FY2018-FY2024. This is the single strongest driver of processing time and the
   reason random k-fold validation flatters a model here.

4. **Audits are latent.** Real disclosure files have no "was this audited"
   column — you infer it from a bimodal processing-time distribution. Audit
   status is generated here as a hidden driver of both processing time and
   outcome, and is NOT written to the CSV unless ``--include-ground-truth`` is
   passed (that flag exists for pipeline validation and leaks the answer, so it
   is off by default).

5. **Wage units.** ``PW_AMOUNT_9089`` and ``WAGE_OFFER_FROM_9089`` are stated in
   the unit named by their own ``*_UNIT_OF_PAY_9089`` column — Year, Hour,
   Month, Week, or Bi-Weekly. A raw mean over the wage column without
   annualising is meaningless, and in ~2% of rows the prevailing-wage unit and
   the offered-wage unit disagree.

6. **Structural messiness** (``--no-messy`` disables): date format changes at
   FY2021 (``MM/DD/YYYY`` -> ``YYYY-MM-DD``), the same employer appears under
   several name spellings, ``CASE_STATUS`` casing varies, worksite state is
   sometimes a full name instead of a 2-letter code, and some string fields
   carry stray whitespace.

7. **Missingness** follows conditional rules, not uniform noise:
   ``ORIG_FILE_DATE`` is populated only when ``REFILE = Y``,
   ``JOB_INFO_ALT_OCC_NUM_MONTHS`` only when ``JOB_INFO_ALT_OCC = Y``,
   attorney fields are blank for self-filed cases, and so on.

Sensitive-attribute note
------------------------
``COUNTRY_OF_CITIZENSHIP`` and ``FW_INFO_BIRTH_COUNTRY`` are genuine fields in
the real disclosure data and are reproduced here. They are far more directly
protected-class-adjacent than the structural proxies (employer size, occupation,
region) named in the project README. Decide deliberately whether they belong in
the feature set, the fairness audit's grouping attributes, or neither — and say
which in the write-up.

Leakage note
------------
``DECISION_DATE`` defines the regression target and must never be a feature.
``CASE_STATUS`` is the classification target. ``PROCESSING_DAYS`` is not written
to the file precisely because it would be the answer; derive it in the prep
layer from the two date columns.

Reproducibility
---------------
Output is a deterministic function of ``--seed`` (default 42), ``--rows``,
``--employers``, and ``--years``. The seed is the provenance record for a
synthetic file the way a download date is for a real one — quote it when
reporting anything built on this data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GENERATOR_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "perm_synthetic_sample.csv"

DEFAULT_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)


# ---------------------------------------------------------------------------
# Fiscal-year regimes
# ---------------------------------------------------------------------------
# Median processing days for a NON-audited case decided in that fiscal year.
# Invented but shaped like the real story: steady pre-COVID, a pandemic bulge,
# a severe FY2022-FY2023 backlog, partial recovery in FY2024.
FY_BACKLOG_MEDIAN_DAYS = {
    2018: 150,
    2019: 168,
    2020: 196,
    2021: 240,
    2022: 330,
    2023: 385,
    2024: 300,
}

# Relative share of rows per fiscal year (normalised over the years requested).
FY_VOLUME_WEIGHT = {
    2018: 0.135,
    2019: 0.145,
    2020: 0.125,
    2021: 0.140,
    2022: 0.155,
    2023: 0.160,
    2024: 0.140,
}

# Extra denial pressure by year (logit units) — audit intensity shifts over time.
FY_DENIAL_SHIFT = {
    2018: 0.00,
    2019: 0.05,
    2020: 0.25,
    2021: 0.20,
    2022: -0.05,
    2023: -0.10,
    2024: 0.05,
}

# Annual wage drift relative to FY2018.
WAGE_INFLATION_PER_YEAR = 0.031


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
# ~26 jurisdictions covering the overwhelming majority of real PERM volume.
STATES = {
    # abbrev: (full name, share, zip2 prefix, wage multiplier, [cities])
    "CA": ("CALIFORNIA", 0.190, "9", 1.18, ["SAN JOSE", "SAN FRANCISCO", "SANTA CLARA", "LOS ANGELES", "SAN DIEGO"]),
    "TX": ("TEXAS", 0.100, "7", 0.98, ["AUSTIN", "DALLAS", "HOUSTON", "PLANO", "IRVING"]),
    "NY": ("NEW YORK", 0.085, "1", 1.15, ["NEW YORK", "BROOKLYN", "ALBANY", "ROCHESTER"]),
    "NJ": ("NEW JERSEY", 0.075, "0", 1.08, ["JERSEY CITY", "EDISON", "PRINCETON", "NEWARK"]),
    "WA": ("WASHINGTON", 0.055, "9", 1.12, ["SEATTLE", "REDMOND", "BELLEVUE", "SPOKANE"]),
    "MA": ("MASSACHUSETTS", 0.045, "0", 1.12, ["BOSTON", "CAMBRIDGE", "WALTHAM", "WORCESTER"]),
    "IL": ("ILLINOIS", 0.040, "6", 1.03, ["CHICAGO", "NAPERVILLE", "SCHAUMBURG"]),
    "GA": ("GEORGIA", 0.035, "3", 0.95, ["ATLANTA", "ALPHARETTA", "SAVANNAH"]),
    "VA": ("VIRGINIA", 0.033, "2", 1.05, ["ARLINGTON", "RESTON", "RICHMOND", "MCLEAN"]),
    "FL": ("FLORIDA", 0.032, "3", 0.93, ["MIAMI", "ORLANDO", "TAMPA", "JACKSONVILLE"]),
    "PA": ("PENNSYLVANIA", 0.025, "1", 0.98, ["PHILADELPHIA", "PITTSBURGH", "MALVERN"]),
    "NC": ("NORTH CAROLINA", 0.025, "2", 0.95, ["CHARLOTTE", "RALEIGH", "DURHAM"]),
    "MI": ("MICHIGAN", 0.020, "4", 0.94, ["DETROIT", "ANN ARBOR", "TROY"]),
    "OH": ("OHIO", 0.018, "4", 0.92, ["COLUMBUS", "CLEVELAND", "CINCINNATI"]),
    "MD": ("MARYLAND", 0.018, "2", 1.05, ["BALTIMORE", "ROCKVILLE", "BETHESDA"]),
    "AZ": ("ARIZONA", 0.017, "8", 0.95, ["PHOENIX", "TEMPE", "CHANDLER"]),
    "CO": ("COLORADO", 0.017, "8", 1.03, ["DENVER", "BOULDER", "COLORADO SPRINGS"]),
    "MN": ("MINNESOTA", 0.015, "5", 1.00, ["MINNEAPOLIS", "SAINT PAUL", "ROCHESTER"]),
    "CT": ("CONNECTICUT", 0.013, "0", 1.06, ["STAMFORD", "HARTFORD", "NEW HAVEN"]),
    "MO": ("MISSOURI", 0.012, "6", 0.91, ["SAINT LOUIS", "KANSAS CITY", "COLUMBIA"]),
    "OR": ("OREGON", 0.012, "9", 1.02, ["PORTLAND", "HILLSBORO", "BEAVERTON"]),
    "WI": ("WISCONSIN", 0.011, "5", 0.93, ["MADISON", "MILWAUKEE", "WAUKESHA"]),
    "TN": ("TENNESSEE", 0.011, "3", 0.90, ["NASHVILLE", "MEMPHIS", "KNOXVILLE"]),
    "IN": ("INDIANA", 0.010, "4", 0.90, ["INDIANAPOLIS", "CARMEL", "BLOOMINGTON"]),
    "UT": ("UTAH", 0.010, "8", 0.96, ["SALT LAKE CITY", "LEHI", "PROVO"]),
    "DC": ("DISTRICT OF COLUMBIA", 0.008, "2", 1.12, ["WASHINGTON"]),
}


# ---------------------------------------------------------------------------
# Employer sectors
# ---------------------------------------------------------------------------
# size_mu/size_sigma parameterise a lognormal headcount, so the pool spans
# 20-person startups through 200k-employee IT services firms.
SECTORS = {
    "TECH": dict(weight=0.22, size_mu=6.2, size_sigma=1.9,
                 naics=["541511", "511210", "518210"]),
    "IT_SERVICES": dict(weight=0.20, size_mu=8.6, size_sigma=1.6,
                        naics=["541512", "541513", "541519"]),
    "FINANCE": dict(weight=0.12, size_mu=7.5, size_sigma=1.8,
                    naics=["522110", "523930", "524113"]),
    "HEALTHCARE": dict(weight=0.13, size_mu=7.2, size_sigma=1.7,
                       naics=["622110", "621111", "621399"]),
    "UNIVERSITY": dict(weight=0.08, size_mu=8.2, size_sigma=1.0,
                       naics=["611310"]),
    "MANUFACTURING": dict(weight=0.11, size_mu=6.8, size_sigma=1.7,
                          naics=["334413", "336111", "325412"]),
    "RESEARCH": dict(weight=0.07, size_mu=5.5, size_sigma=1.5,
                     naics=["541715", "541714"]),
    "RETAIL_HOSPITALITY": dict(weight=0.07, size_mu=5.8, size_sigma=1.8,
                               naics=["722511", "445110", "452319"]),
}


# ---------------------------------------------------------------------------
# Occupations
# ---------------------------------------------------------------------------
# `code10`/`title10` are 2010 SOC (used for FY2018-FY2019 rows);
# `code18`/`title18` are 2018 SOC (FY2020+). Several 2010 codes collapse into
# one 2018 code — the classic many-to-one drift that breaks naive encoders.
# `base` is a Level-II annual prevailing wage in FY2018 dollars.
# `sectors` weights which employers file for the occupation.
OCCUPATIONS = [
    dict(code10="15-1132", title10="Software Developers, Applications",
         code18="15-1252", title18="Software Developers", base=108000,
         titles=["Software Engineer", "Senior Software Engineer", "Software Developer",
                 "Applications Developer", "Full Stack Engineer"],
         sectors=dict(TECH=34, IT_SERVICES=26, FINANCE=12, RESEARCH=5,
                      MANUFACTURING=5, HEALTHCARE=3, RETAIL_HOSPITALITY=3, UNIVERSITY=1)),
    dict(code10="15-1133", title10="Software Developers, Systems Software",
         code18="15-1252", title18="Software Developers", base=118000,
         titles=["Systems Software Engineer", "Platform Engineer", "Senior Systems Engineer",
                 "Embedded Software Engineer"],
         sectors=dict(TECH=16, IT_SERVICES=8, FINANCE=4, MANUFACTURING=6, RESEARCH=3)),
    dict(code10="15-1121", title10="Computer Systems Analysts",
         code18="15-1211", title18="Computer Systems Analysts", base=92000,
         titles=["Systems Analyst", "Business Systems Analyst", "IT Analyst",
                 "Senior Systems Analyst"],
         sectors=dict(IT_SERVICES=18, TECH=6, FINANCE=10, HEALTHCARE=6,
                      MANUFACTURING=4, RETAIL_HOSPITALITY=4, UNIVERSITY=2)),
    dict(code10="15-1131", title10="Computer Programmers",
         code18="15-1251", title18="Computer Programmers", base=84000,
         titles=["Programmer Analyst", "Applications Programmer", "Software Programmer"],
         sectors=dict(IT_SERVICES=12, TECH=3, FINANCE=4, HEALTHCARE=3, MANUFACTURING=2)),
    dict(code10="15-1199", title10="Computer Occupations, All Other",
         code18="15-1299", title18="Computer Occupations, All Other", base=96000,
         titles=["Technical Consultant", "DevOps Engineer", "Cloud Engineer",
                 "Site Reliability Engineer"],
         sectors=dict(IT_SERVICES=10, TECH=8, FINANCE=5, RESEARCH=3, MANUFACTURING=3)),
    dict(code10="15-1142", title10="Network and Computer Systems Administrators",
         code18="15-1244", title18="Network and Computer Systems Administrators", base=82000,
         titles=["Systems Administrator", "Network Administrator", "Infrastructure Engineer"],
         sectors=dict(IT_SERVICES=6, TECH=3, UNIVERSITY=3, HEALTHCARE=3, FINANCE=2)),
    dict(code10="15-1141", title10="Database Administrators",
         code18="15-1242", title18="Database Administrators", base=90000,
         titles=["Database Administrator", "Senior DBA", "Data Engineer"],
         sectors=dict(IT_SERVICES=5, TECH=4, FINANCE=4, HEALTHCARE=2, UNIVERSITY=1)),
    dict(code10="15-2031", title10="Operations Research Analysts",
         code18="15-2031", title18="Operations Research Analysts", base=88000,
         titles=["Operations Research Analyst", "Quantitative Analyst", "Decision Scientist"],
         sectors=dict(FINANCE=6, TECH=3, MANUFACTURING=3, RESEARCH=3, RETAIL_HOSPITALITY=2)),
    dict(code10="15-2041", title10="Statisticians",
         code18="15-2051", title18="Data Scientists", base=102000,
         titles=["Data Scientist", "Senior Data Scientist", "Machine Learning Engineer",
                 "Applied Scientist"],
         sectors=dict(TECH=9, RESEARCH=6, FINANCE=5, HEALTHCARE=3, IT_SERVICES=3, UNIVERSITY=2)),
    dict(code10="11-3021", title10="Computer and Information Systems Managers",
         code18="11-3021", title18="Computer and Information Systems Managers", base=145000,
         titles=["IT Manager", "Engineering Manager", "Director of Engineering",
                 "Senior Technical Manager"],
         sectors=dict(TECH=7, IT_SERVICES=7, FINANCE=4, HEALTHCARE=3, MANUFACTURING=3)),
    dict(code10="11-1021", title10="General and Operations Managers",
         code18="11-1021", title18="General and Operations Managers", base=112000,
         titles=["Operations Manager", "General Manager", "Business Operations Manager"],
         sectors=dict(RETAIL_HOSPITALITY=8, MANUFACTURING=5, IT_SERVICES=3,
                      HEALTHCARE=3, FINANCE=3)),
    dict(code10="11-2021", title10="Marketing Managers",
         code18="11-2021", title18="Marketing Managers", base=125000,
         titles=["Marketing Manager", "Product Marketing Manager", "Brand Manager"],
         sectors=dict(TECH=4, RETAIL_HOSPITALITY=3, FINANCE=2, MANUFACTURING=2)),
    dict(code10="11-3031", title10="Financial Managers",
         code18="11-3031", title18="Financial Managers", base=130000,
         titles=["Finance Manager", "Controller", "Treasury Manager"],
         sectors=dict(FINANCE=7, MANUFACTURING=2, HEALTHCARE=2, RETAIL_HOSPITALITY=2)),
    dict(code10="13-2011", title10="Accountants and Auditors",
         code18="13-2011", title18="Accountants and Auditors", base=72000,
         titles=["Accountant", "Senior Accountant", "Financial Auditor", "Tax Accountant"],
         sectors=dict(FINANCE=8, RETAIL_HOSPITALITY=3, MANUFACTURING=3,
                      IT_SERVICES=2, HEALTHCARE=2)),
    dict(code10="13-1111", title10="Management Analysts",
         code18="13-1111", title18="Management Analysts", base=92000,
         titles=["Management Consultant", "Business Analyst", "Strategy Consultant"],
         sectors=dict(IT_SERVICES=9, FINANCE=5, HEALTHCARE=3, MANUFACTURING=2, TECH=2)),
    dict(code10="13-2051", title10="Financial Analysts",
         code18="13-2051", title18="Financial and Investment Analysts", base=95000,
         titles=["Financial Analyst", "Senior Financial Analyst", "Investment Analyst"],
         sectors=dict(FINANCE=9, MANUFACTURING=2, TECH=2, RETAIL_HOSPITALITY=1)),
    dict(code10="13-1161", title10="Market Research Analysts and Marketing Specialists",
         code18="13-1161", title18="Market Research Analysts and Marketing Specialists",
         base=76000,
         titles=["Market Research Analyst", "Marketing Specialist", "Growth Analyst"],
         sectors=dict(TECH=3, RETAIL_HOSPITALITY=3, FINANCE=2, IT_SERVICES=2)),
    dict(code10="17-2071", title10="Electrical Engineers",
         code18="17-2071", title18="Electrical Engineers", base=102000,
         titles=["Electrical Engineer", "Senior Electrical Engineer", "Hardware Design Engineer"],
         sectors=dict(MANUFACTURING=10, TECH=4, RESEARCH=3)),
    dict(code10="17-2141", title10="Mechanical Engineers",
         code18="17-2141", title18="Mechanical Engineers", base=95000,
         titles=["Mechanical Engineer", "Senior Mechanical Engineer", "Design Engineer"],
         sectors=dict(MANUFACTURING=11, RESEARCH=3, TECH=2)),
    dict(code10="17-2112", title10="Industrial Engineers",
         code18="17-2112", title18="Industrial Engineers", base=90000,
         titles=["Industrial Engineer", "Process Engineer", "Manufacturing Engineer"],
         sectors=dict(MANUFACTURING=9, RETAIL_HOSPITALITY=2, RESEARCH=1)),
    dict(code10="17-2061", title10="Computer Hardware Engineers",
         code18="17-2061", title18="Computer Hardware Engineers", base=118000,
         titles=["Hardware Engineer", "ASIC Design Engineer", "Firmware Engineer"],
         sectors=dict(TECH=5, MANUFACTURING=5, RESEARCH=2)),
    dict(code10="17-2051", title10="Civil Engineers",
         code18="17-2051", title18="Civil Engineers", base=86000,
         titles=["Civil Engineer", "Structural Engineer", "Project Engineer"],
         sectors=dict(MANUFACTURING=4, RESEARCH=1)),
    dict(code10="29-1141", title10="Registered Nurses",
         code18="29-1141", title18="Registered Nurses", base=72000,
         titles=["Registered Nurse", "Staff Nurse", "Clinical Nurse"],
         sectors=dict(HEALTHCARE=20)),
    dict(code10="29-1062", title10="Family and General Practitioners",
         code18="29-1215", title18="Family Medicine Physicians", base=205000,
         titles=["Family Medicine Physician", "Primary Care Physician", "Staff Physician"],
         sectors=dict(HEALTHCARE=9, UNIVERSITY=2)),
    dict(code10="29-1069", title10="Physicians and Surgeons, All Other",
         code18="29-1229", title18="Physicians, All Other", base=230000,
         titles=["Physician", "Attending Physician", "Hospitalist", "Specialist Physician"],
         sectors=dict(HEALTHCARE=10, UNIVERSITY=3, RESEARCH=1)),
    dict(code10="29-1051", title10="Pharmacists",
         code18="29-1051", title18="Pharmacists", base=118000,
         titles=["Pharmacist", "Clinical Pharmacist", "Staff Pharmacist"],
         sectors=dict(HEALTHCARE=6, RETAIL_HOSPITALITY=2)),
    dict(code10="29-1123", title10="Physical Therapists",
         code18="29-1123", title18="Physical Therapists", base=84000,
         titles=["Physical Therapist", "Senior Physical Therapist"],
         sectors=dict(HEALTHCARE=5)),
    dict(code10="19-1042", title10="Medical Scientists, Except Epidemiologists",
         code18="19-1042", title18="Medical Scientists, Except Epidemiologists", base=88000,
         titles=["Research Scientist", "Postdoctoral Research Associate", "Medical Scientist"],
         sectors=dict(RESEARCH=12, UNIVERSITY=8, HEALTHCARE=3)),
    dict(code10="19-2031", title10="Chemists",
         code18="19-2031", title18="Chemists", base=78000,
         titles=["Chemist", "Analytical Chemist", "Research Chemist"],
         sectors=dict(RESEARCH=7, MANUFACTURING=4, UNIVERSITY=2)),
    dict(code10="19-3011", title10="Economists",
         code18="19-3011", title18="Economists", base=105000,
         titles=["Economist", "Research Economist", "Senior Economist"],
         sectors=dict(RESEARCH=3, FINANCE=3, UNIVERSITY=2)),
    dict(code10="25-1071", title10="Health Specialties Teachers, Postsecondary",
         code18="25-1071", title18="Health Specialties Teachers, Postsecondary", base=95000,
         titles=["Assistant Professor", "Associate Professor", "Clinical Assistant Professor"],
         sectors=dict(UNIVERSITY=14, HEALTHCARE=2)),
    dict(code10="25-1022", title10="Mathematical Science Teachers, Postsecondary",
         code18="25-1022", title18="Mathematical Science Teachers, Postsecondary", base=80000,
         titles=["Assistant Professor of Mathematics", "Lecturer", "Associate Professor"],
         sectors=dict(UNIVERSITY=10)),
    dict(code10="25-1042", title10="Biological Science Teachers, Postsecondary",
         code18="25-1042", title18="Biological Science Teachers, Postsecondary", base=82000,
         titles=["Assistant Professor of Biology", "Research Assistant Professor"],
         sectors=dict(UNIVERSITY=9, RESEARCH=2)),
    dict(code10="25-2021", title10="Elementary School Teachers, Except Special Education",
         code18="25-2021", title18="Elementary School Teachers, Except Special Education",
         base=58000,
         titles=["Elementary School Teacher", "Bilingual Teacher", "Classroom Teacher"],
         sectors=dict(UNIVERSITY=4)),
    dict(code10="27-1024", title10="Graphic Designers",
         code18="27-1024", title18="Graphic Designers", base=62000,
         titles=["Graphic Designer", "UX Designer", "Visual Designer", "Product Designer"],
         sectors=dict(TECH=3, RETAIL_HOSPITALITY=2, IT_SERVICES=1)),
    dict(code10="41-3099", title10="Sales Representatives, Services, All Other",
         code18="41-3091", title18="Sales Representatives of Services, All Other", base=68000,
         titles=["Account Executive", "Sales Representative", "Business Development Manager"],
         sectors=dict(RETAIL_HOSPITALITY=5, TECH=3, IT_SERVICES=3, FINANCE=2)),
    dict(code10="35-1011", title10="Chefs and Head Cooks",
         code18="35-1011", title18="Chefs and Head Cooks", base=52000,
         titles=["Chef", "Executive Chef", "Head Cook", "Sous Chef"],
         sectors=dict(RETAIL_HOSPITALITY=9)),
    dict(code10="51-1011", title10="First-Line Supervisors of Production and Operating Workers",
         code18="51-1011", title18="First-Line Supervisors of Production and Operating Workers",
         base=64000,
         titles=["Production Supervisor", "Manufacturing Supervisor", "Shift Supervisor"],
         sectors=dict(MANUFACTURING=6, RETAIL_HOSPITALITY=2)),
]


# ---------------------------------------------------------------------------
# Categorical vocabularies (values match real disclosure-file spellings)
# ---------------------------------------------------------------------------
CASE_STATUSES = ["Certified", "Denied", "Withdrawn", "Certified-Expired"]

SKILL_LEVELS = ["Level I", "Level II", "Level III", "Level IV"]
SKILL_LEVEL_P = [0.12, 0.42, 0.30, 0.16]
SKILL_LEVEL_WAGE_MULT = {"Level I": 0.76, "Level II": 0.92, "Level III": 1.06, "Level IV": 1.24}

PW_SOURCES = ["OES", "CBA", "DBA", "SCA", "Other"]
PW_SOURCE_P = [0.945, 0.020, 0.010, 0.010, 0.015]

UNITS_OF_PAY = ["Year", "Hour", "Month", "Week", "Bi-Weekly"]
UNIT_P = [0.865, 0.098, 0.017, 0.010, 0.010]
UNIT_DIVISOR = {"Year": 1.0, "Hour": 2080.0, "Month": 12.0, "Week": 52.0, "Bi-Weekly": 26.0}

EDUCATION_LEVELS = ["None", "High School", "Associate's", "Bachelor's",
                    "Master's", "Doctorate", "Other"]
EDUCATION_P = [0.012, 0.030, 0.018, 0.492, 0.352, 0.081, 0.015]
EDUCATION_RANK = {"None": 0, "High School": 1, "Other": 1, "Associate's": 2,
                  "Bachelor's": 3, "Master's": 4, "Doctorate": 5}

MAJORS = ["COMPUTER SCIENCE", "ELECTRICAL ENGINEERING", "COMPUTER ENGINEERING",
          "INFORMATION TECHNOLOGY", "MECHANICAL ENGINEERING", "BUSINESS ADMINISTRATION",
          "MATHEMATICS", "STATISTICS", "FINANCE", "ACCOUNTING", "NURSING", "BIOLOGY",
          "CHEMISTRY", "PHYSICS", "ECONOMICS", "MEDICINE", "PHARMACY", "MARKETING",
          "INDUSTRIAL ENGINEERING", "CIVIL ENGINEERING", "ANY ENGINEERING FIELD",
          "COMPUTER SCIENCE OR RELATED", "INFORMATION SYSTEMS"]

CLASS_OF_ADMISSION = ["H-1B", "L-1", "F-1", "Not in USA", "TN", "H-4", "E-2",
                      "O-1", "J-1", "L-2", "B-2", "H-1B1", "Parolee", "E-3"]
CLASS_P = [0.640, 0.058, 0.092, 0.086, 0.026, 0.024, 0.012,
           0.014, 0.014, 0.014, 0.006, 0.006, 0.004, 0.004]

COUNTRIES = ["INDIA", "CHINA", "SOUTH KOREA", "CANADA", "MEXICO", "PHILIPPINES",
             "BRAZIL", "TAIWAN", "UNITED KINGDOM", "NEPAL", "PAKISTAN", "TURKEY",
             "NIGERIA", "VENEZUELA", "COLOMBIA", "JAPAN", "FRANCE", "GERMANY",
             "RUSSIA", "VIETNAM", "IRAN", "SPAIN", "ITALY", "ARGENTINA",
             "BANGLADESH", "SRI LANKA", "UKRAINE", "AUSTRALIA", "ISRAEL", "EGYPT"]
COUNTRY_P = [0.585, 0.108, 0.036, 0.030, 0.024, 0.021, 0.016, 0.015, 0.013,
             0.012, 0.011, 0.010, 0.009, 0.008, 0.008, 0.007, 0.006, 0.006,
             0.006, 0.006, 0.005, 0.005, 0.005, 0.004, 0.004, 0.004, 0.004,
             0.003, 0.003, 0.006]

PREPARER_TITLES = ["Attorney", "Immigration Specialist", "HR Manager", "Paralegal",
                   "Human Resources Director", "Global Mobility Manager", "Owner"]
PREPARER_P = [0.52, 0.14, 0.13, 0.09, 0.06, 0.04, 0.02]

ATTORNEY_FIRMS = [
    "FRAGOMEN DEL REY", "BERRY APPLEMAN", "OGLETREE DEAKINS", "SEYFARTH SHAW",
    "GREENBERG TRAURIG", "JACKSON LEWIS", "MINTZ LEVIN", "ERICKSON IMMIGRATION",
    "CHUGH LLP", "MURTHY LAW FIRM", "REDDY NEUMANN", "VISANOW LEGAL",
    "CLARK HILL", "LITTLER MENDELSON", "MAGGIO KATTAR", "SHIHAB BURKE",
]

# Employer-name building blocks.
NAME_STEMS = [
    "NEXORA", "VANTIVA", "BLUEPEAK", "CORTEXA", "HELIOSTAR", "NORTHWIND", "AXIOMEDGE",
    "CLARIVAULT", "SUMMITRY", "QUANTIFI", "BRIGHTPATH", "IRONBRIDGE", "CEDARLINE",
    "PACIFICORE", "ORBITAL", "LUMENFIELD", "STRATOSPHERE", "ARCADIAN", "MERIDIAN",
    "KEYSTONE", "SILVERLAKE", "TRUENORTH", "EVERGREEN", "REDWOOD", "HARBORVIEW",
    "GRANITEPOINT", "WHITEOAK", "BLACKSTONE RIDGE", "COPPERFIELD", "STARLING",
    "ATLASWORKS", "CINDERPEAK", "VERIDIAN", "NOVATERRA", "PINNACLE", "CASCADIA",
    "FIRSTLIGHT", "OAKMONT", "TITANFORGE", "ZENITHRA", "CALDERA", "HALCYON",
    "MAGNOLIA", "SENTINEL", "AURELIA", "BRIARWOOD", "CROSSWIND", "DELTAWORKS",
    "EASTGATE", "FAIRHAVEN", "GOLDENROD", "HAWTHORNE", "INLETPOINT", "JUNIPER",
]

SECTOR_SUFFIXES = {
    "TECH": ["TECHNOLOGIES INC", "SOFTWARE INC", "LABS INC", "SYSTEMS INC", "AI INC"],
    "IT_SERVICES": ["CONSULTING SERVICES INC", "IT SOLUTIONS INC", "INFOTECH INC",
                    "TECHNOLOGY SERVICES LLC", "GLOBAL SERVICES INC"],
    "FINANCE": ["FINANCIAL GROUP", "CAPITAL PARTNERS", "BANK NA", "ASSET MANAGEMENT LLC",
                "INSURANCE GROUP"],
    "HEALTHCARE": ["HEALTH SYSTEM", "MEDICAL CENTER", "HOSPITAL", "HEALTHCARE PARTNERS",
                   "CLINICS INC"],
    "MANUFACTURING": ["INDUSTRIES INC", "MANUFACTURING CO", "MOTORS CORP",
                      "AEROSPACE INC", "MATERIALS CORP"],
    "RESEARCH": ["RESEARCH INSTITUTE", "BIOSCIENCES INC", "THERAPEUTICS INC",
                 "LABORATORIES INC", "SCIENCES CORP"],
    "RETAIL_HOSPITALITY": ["RETAIL GROUP INC", "HOSPITALITY GROUP", "RESTAURANTS INC",
                           "MARKETS LLC", "STORES INC"],
}

UNIVERSITY_PATTERNS = [
    "UNIVERSITY OF {stem}", "{stem} STATE UNIVERSITY", "{stem} INSTITUTE OF TECHNOLOGY",
    "{stem} COLLEGE", "{stem} UNIVERSITY",
]

# Inserted before the suffix to keep employer names unique without resorting to
# a trailing integer, which no real corporate name carries.
NAME_QUALIFIERS = [
    "USA", "NORTH AMERICA", "INTERNATIONAL", "HOLDINGS", "GLOBAL", "PARTNERS",
    "ENTERPRISES", "AMERICAS", "PACIFIC", "ATLANTIC", "MIDWEST", "NORTHEAST",
    "SOUTHWEST", "WEST", "EAST", "CENTRAL", "ADVANCED", "APPLIED", "UNITED",
    "PREMIER",
]

STREET_TYPES = ["STREET", "AVENUE", "BOULEVARD", "DRIVE", "ROAD", "PARKWAY", "WAY"]
STREET_NAMES = ["MAIN", "PARK", "TECHNOLOGY", "INNOVATION", "MARKET", "WASHINGTON",
                "CORPORATE", "RESEARCH", "COMMERCE", "UNIVERSITY", "LINCOLN", "BRIDGE"]


# Columns whose order mirrors a real PERM disclosure workbook.
COLUMN_ORDER = [
    "CASE_NUMBER", "CASE_STATUS", "RECEIVED_DATE", "DECISION_DATE", "FISCAL_YEAR",
    "REFILE", "ORIG_FILE_DATE", "SCHD_A_SHEEPHERDER",
    "EMPLOYER_NAME", "EMPLOYER_ADDRESS_1", "EMPLOYER_CITY", "EMPLOYER_STATE",
    "EMPLOYER_POSTAL_CODE", "EMPLOYER_COUNTRY", "EMPLOYER_NUM_EMPLOYEES",
    "EMPLOYER_YEAR_COMMENCED_BUSINESS", "NAICS_CODE", "FW_OWNERSHIP_INTEREST",
    "PW_SOC_CODE", "PW_SOC_TITLE", "PW_SKILL_LEVEL", "PW_SOURCE", "PW_TRACK_NUM",
    "PW_DETERM_DATE", "PW_EXPIRE_DATE", "PW_AMOUNT_9089", "PW_UNIT_OF_PAY_9089",
    "WAGE_OFFER_FROM_9089", "WAGE_OFFER_TO_9089", "WAGE_OFFER_UNIT_OF_PAY_9089",
    "JOB_INFO_JOB_TITLE", "JOB_INFO_WORK_CITY", "JOB_INFO_WORK_STATE",
    "JOB_INFO_WORK_POSTAL_CODE", "JOB_INFO_EDUCATION", "JOB_INFO_MAJOR",
    "JOB_INFO_TRAINING", "JOB_INFO_EXPERIENCE", "JOB_INFO_EXPERIENCE_NUM_MONTHS",
    "JOB_INFO_ALT_OCC", "JOB_INFO_ALT_OCC_NUM_MONTHS", "JOB_INFO_JOB_REQ_NORMAL",
    "JOB_INFO_FOREIGN_LANG_REQ", "JOB_INFO_COMBO_OCCUPATION",
    "RECR_INFO_PROFESSIONAL_OCC", "RECR_INFO_COLL_UNIV_TEACHER",
    "RECR_INFO_SUNDAY_NEWSPAPER", "RI_LAYOFF_IN_PAST_SIX_MONTHS",
    "FW_INFO_BIRTH_COUNTRY", "COUNTRY_OF_CITIZENSHIP", "CLASS_OF_ADMISSION",
    "FOREIGN_WORKER_INFO_EDUCATION", "FW_INFO_YR_REL_EDU_COMPLETED",
    "FW_INFO_REQ_EXPERIENCE", "AGENT_ATTORNEY_NAME", "AGENT_ATTORNEY_CITY",
    "AGENT_ATTORNEY_STATE", "PREPARER_INFO_TITLE",
]

GROUND_TRUTH_COLUMNS = ["_SYNTH_AUDITED", "_SYNTH_PROCESSING_DAYS", "_SYNTH_SECTOR"]


@dataclass
class GeneratorConfig:
    """Knobs for a generation run. Output is deterministic given these."""

    n_rows: int = 50_000
    n_employers: int = 1_400
    years: tuple[int, ...] = DEFAULT_YEARS
    seed: int = 42
    messy: bool = True
    include_ground_truth: bool = False
    output_path: Path = field(default_factory=lambda: DEFAULT_OUTPUT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fiscal_year_bounds(fy: int) -> tuple[np.datetime64, np.datetime64]:
    """Federal fiscal year FY runs 1 Oct (FY-1) through 30 Sep (FY)."""
    return np.datetime64(f"{fy - 1}-10-01"), np.datetime64(f"{fy}-09-30")


def _normalise(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def _allocate_rows(rng: np.random.Generator, n_rows: int,
                   years: tuple[int, ...]) -> np.ndarray:
    """Split n_rows across fiscal years, leftover going to the largest years."""
    w = _normalise(np.array([FY_VOLUME_WEIGHT.get(y, 0.14) for y in years], dtype=float))
    counts = np.floor(w * n_rows).astype(int)
    while counts.sum() < n_rows:
        counts[int(np.argmax(w - counts / max(counts.sum(), 1)))] += 1
    return counts


def _unique_employer_name(rng: np.random.Generator, sector: str, city: str,
                          used: set[str]) -> str:
    """Draw a plausible employer name not already in ``used``.

    Escalates through decorations rather than appending a counter, so every name
    still reads like a company (or a university) after de-duplication.
    """
    for attempt in range(400):
        stem = NAME_STEMS[int(rng.integers(len(NAME_STEMS)))]
        if sector == "UNIVERSITY":
            pattern = UNIVERSITY_PATTERNS[int(rng.integers(len(UNIVERSITY_PATTERNS)))]
            name = pattern.format(stem=stem)
            if attempt >= 2:
                name = f"{name} AT {city}"
        else:
            suffix = SECTOR_SUFFIXES[sector][int(rng.integers(len(SECTOR_SUFFIXES[sector])))]
            if attempt < 2:
                name = f"{stem} {suffix}"
            else:
                qual = NAME_QUALIFIERS[int(rng.integers(len(NAME_QUALIFIERS)))]
                name = f"{stem} {qual} {suffix}"
        if name not in used:
            used.add(name)
            return name
    # Exhausted the name space for this sector — widen it with a second stem.
    while True:
        a = NAME_STEMS[int(rng.integers(len(NAME_STEMS)))]
        b = NAME_STEMS[int(rng.integers(len(NAME_STEMS)))]
        if sector == "UNIVERSITY":
            name = f"{a}-{b} UNIVERSITY"
        else:
            name = f"{a}-{b} {SECTOR_SUFFIXES[sector][0]}"
        if name not in used:
            used.add(name)
            return name


def _build_employer_pool(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Employers with a Zipf-ish filing-volume distribution.

    A handful of large IT-services firms account for a large share of filings,
    which is both true of real PERM data and the reason employer-level features
    need care: the top employers dominate any unweighted average.
    """
    sector_names = list(SECTORS)
    sector_p = _normalise(np.array([SECTORS[s]["weight"] for s in sector_names]))
    sectors = rng.choice(sector_names, size=n, p=sector_p)

    abbrevs = list(STATES)
    state_p = _normalise(np.array([STATES[s][1] for s in abbrevs]))
    hq_state = rng.choice(abbrevs, size=n, p=state_p)
    hq_city = np.array([
        STATES[s][4][int(rng.integers(len(STATES[s][4])))] for s in hq_state
    ], dtype=object)

    names = np.empty(n, dtype=object)
    used: set[str] = set()
    for i in range(n):
        names[i] = _unique_employer_name(rng, sectors[i], hq_city[i], used)

    size_mu = np.array([SECTORS[s]["size_mu"] for s in sectors])
    size_sigma = np.array([SECTORS[s]["size_sigma"] for s in sectors])
    num_employees = np.clip(
        np.round(rng.lognormal(size_mu, size_sigma)).astype(int), 3, 420_000
    )

    naics = np.array([
        SECTORS[s]["naics"][int(rng.integers(len(SECTORS[s]["naics"])))] for s in sectors
    ], dtype=object)

    year_commenced = np.clip(
        np.round(2020 - rng.gamma(shape=2.2, scale=11.0, size=n)).astype(int), 1890, 2023
    )

    # Filing volume: Zipf-like over a shuffled rank so volume is independent of
    # the employer's position in the array.
    ranks = rng.permutation(n) + 1
    filing_weight = _normalise(1.0 / np.power(ranks, 0.92))

    address = np.array([
        f"{int(rng.integers(1, 9999))} "
        f"{STREET_NAMES[int(rng.integers(len(STREET_NAMES)))]} "
        f"{STREET_TYPES[int(rng.integers(len(STREET_TYPES)))]}"
        for _ in range(n)
    ], dtype=object)

    return dict(
        name=names, sector=sectors, num_employees=num_employees, naics=naics,
        state=hq_state, city=hq_city, year_commenced=year_commenced,
        filing_weight=filing_weight, address=address,
    )


def _sector_occupation_matrix() -> tuple[list[str], np.ndarray]:
    """Per-sector probability vector over OCCUPATIONS."""
    sector_names = list(SECTORS)
    mat = np.zeros((len(sector_names), len(OCCUPATIONS)), dtype=float)
    for j, occ in enumerate(OCCUPATIONS):
        for sec, w in occ["sectors"].items():
            mat[sector_names.index(sec), j] = w
    # A sector with no listed occupation falls back to uniform.
    for i in range(mat.shape[0]):
        if mat[i].sum() == 0:
            mat[i] = 1.0
        mat[i] = _normalise(mat[i])
    return sector_names, mat


def _zip_for_state(rng: np.random.Generator, states: np.ndarray) -> np.ndarray:
    prefixes = np.array([STATES[s][2] for s in states])
    tails = rng.integers(0, 10_000, size=len(states))
    return np.array([f"{p}{t:04d}" for p, t in zip(prefixes, tails)], dtype=object)


def _format_dates(values: pd.Series, fy: np.ndarray, messy: bool) -> pd.Series:
    """Write dates as text, with a format change at FY2021 when messy."""
    out = pd.Series(pd.NA, index=values.index, dtype=object)
    valid = values.notna()
    if not messy:
        out[valid] = values[valid].dt.strftime("%Y-%m-%d")
        return out
    early = valid & (pd.Series(fy, index=values.index) <= 2020)
    late = valid & ~early
    out[early] = values[early].dt.strftime("%m/%d/%Y")
    out[late] = values[late].dt.strftime("%Y-%m-%d")
    return out


def _apply_name_variants(rng: np.random.Generator, names: np.ndarray) -> np.ndarray:
    """Same employer, several spellings — the entity-resolution problem in miniature."""
    out = names.astype(object).copy()
    roll = rng.random(len(out))
    for i, r in enumerate(roll):
        if r < 0.06:
            out[i] = str(out[i]).title()
        elif r < 0.10:
            out[i] = str(out[i]).replace(" INC", ", INC.").replace(" LLC", ", LLC")
        elif r < 0.13:
            out[i] = f"  {out[i]} "
        elif r < 0.15:
            out[i] = str(out[i]).title().replace(" Inc", ", Inc.")
    return out


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def generate(config: GeneratorConfig) -> pd.DataFrame:
    """Build the synthetic dataset as a DataFrame."""
    rng = np.random.default_rng(config.seed)
    years = tuple(config.years)
    n = config.n_rows

    employers = _build_employer_pool(rng, config.n_employers)
    sector_names, occ_matrix = _sector_occupation_matrix()

    # --- fiscal year assignment (decision FY) ------------------------------
    counts = _allocate_rows(rng, n, years)
    fy = np.repeat(np.array(years), counts)

    # --- employer assignment ------------------------------------------------
    emp_idx = rng.choice(config.n_employers, size=n, p=employers["filing_weight"])
    emp_sector = employers["sector"][emp_idx]
    # Headcount is self-reported per filing and the company grows, so the same
    # employer reports different numbers across years. Without this drift,
    # EMPLOYER_NUM_EMPLOYEES is a perfect employer fingerprint and a model can
    # memorise employer identity through it.
    emp_size = np.clip(
        np.round(
            employers["num_employees"][emp_idx].astype(float)
            * np.exp(0.035 * (fy - 2021))
            * rng.lognormal(0.0, 0.06, size=n)
        ),
        3, 500_000,
    ).astype(int)

    # --- occupation, conditioned on the employer's sector -------------------
    occ_idx = np.empty(n, dtype=int)
    for si, sname in enumerate(sector_names):
        mask = emp_sector == sname
        k = int(mask.sum())
        if k:
            occ_idx[mask] = rng.choice(len(OCCUPATIONS), size=k, p=occ_matrix[si])

    use_2018_soc = fy >= 2020
    soc_code = np.where(
        use_2018_soc,
        np.array([OCCUPATIONS[i]["code18"] for i in occ_idx], dtype=object),
        np.array([OCCUPATIONS[i]["code10"] for i in occ_idx], dtype=object),
    )
    soc_title = np.where(
        use_2018_soc,
        np.array([OCCUPATIONS[i]["title18"] for i in occ_idx], dtype=object),
        np.array([OCCUPATIONS[i]["title10"] for i in occ_idx], dtype=object),
    )
    job_title = np.array([
        OCCUPATIONS[i]["titles"][int(rng.integers(len(OCCUPATIONS[i]["titles"])))]
        for i in occ_idx
    ], dtype=object)

    # --- worksite: usually but not always the employer's HQ state -----------
    abbrevs = list(STATES)
    state_p = _normalise(np.array([STATES[s][1] for s in abbrevs]))
    work_state = np.where(
        rng.random(n) < 0.72,
        employers["state"][emp_idx],
        rng.choice(abbrevs, size=n, p=state_p),
    )
    work_city = np.array([
        STATES[s][4][int(rng.integers(len(STATES[s][4])))] for s in work_state
    ], dtype=object)

    # --- wages ---------------------------------------------------------------
    skill_level = rng.choice(SKILL_LEVELS, size=n, p=SKILL_LEVEL_P)
    base_wage = np.array([OCCUPATIONS[i]["base"] for i in occ_idx], dtype=float)
    level_mult = np.array([SKILL_LEVEL_WAGE_MULT[s] for s in skill_level])
    geo_mult = np.array([STATES[s][3] for s in work_state])
    year_mult = np.power(1.0 + WAGE_INFLATION_PER_YEAR, fy - 2018)
    noise = rng.lognormal(0.0, 0.11, size=n)

    pw_annual = base_wage * level_mult * geo_mult * year_mult * noise

    # Offered wage sits at or above prevailing in almost every real case; the
    # rare shortfall is a strong denial signal.
    # Meeting the prevailing wage is the legal test, so the premium must be
    # strictly positive. A lognormal centred just above 1.0 will not do: with
    # sigma=0.075 it still drops ~27% of offers under the prevailing wage and
    # drowns the deliberate below_pw signal. Build it as 1 + a positive draw.
    premium = 1.0 + rng.lognormal(-2.60, 0.75, size=n)
    below_pw = rng.random(n) < 0.016
    premium = np.where(below_pw, rng.uniform(0.86, 0.995, size=n), premium)
    offer_annual = pw_annual * premium
    round_it = rng.random(n) < 0.55
    offer_annual = np.where(round_it, np.round(offer_annual / 1000.0) * 1000.0, offer_annual)
    # Rounding to the nearest $1k can nudge an offer back under the prevailing
    # wage; only the deliberate below_pw rows are allowed to sit there.
    offer_annual = np.where(below_pw, offer_annual, np.maximum(offer_annual, pw_annual))

    pw_unit = rng.choice(UNITS_OF_PAY, size=n, p=UNIT_P)
    # ~2% of rows state the offered wage in a unit that differs from the PW's.
    # Rotate the index rather than redrawing, which would usually land on "Year"
    # again and yield a far lower mismatch rate than intended.
    unit_mismatch = rng.random(n) < 0.02
    pw_unit_idx = np.array([UNITS_OF_PAY.index(u) for u in pw_unit])
    rotation = rng.integers(1, len(UNITS_OF_PAY), size=n)
    offer_unit = np.where(
        unit_mismatch,
        np.array(UNITS_OF_PAY, dtype=object)[(pw_unit_idx + rotation) % len(UNITS_OF_PAY)],
        pw_unit,
    )

    pw_div = np.array([UNIT_DIVISOR[u] for u in pw_unit])
    offer_div = np.array([UNIT_DIVISOR[u] for u in offer_unit])
    pw_amount = np.round(pw_annual / pw_div, 2)
    offer_from = np.round(offer_annual / offer_div, 2)
    # WAGE_OFFER_TO is usually blank; when present it tops a stated range.
    has_range = rng.random(n) < 0.24
    offer_to = np.where(has_range, np.round(offer_from * rng.uniform(1.05, 1.35, size=n), 2), np.nan)

    # --- job requirements ----------------------------------------------------
    job_education = rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_P)
    exp_required = rng.random(n) < 0.78
    exp_months = np.where(
        exp_required,
        rng.choice([12, 24, 36, 48, 60, 84], size=n, p=[0.10, 0.36, 0.26, 0.12, 0.12, 0.04]),
        np.nan,
    )
    alt_occ = rng.random(n) < 0.34
    alt_occ_months = np.where(alt_occ, rng.choice([24, 36, 48, 60], size=n,
                                                  p=[0.42, 0.32, 0.14, 0.12]), np.nan)
    foreign_lang = rng.random(n) < 0.048
    combo_occupation = rng.random(n) < 0.082
    job_req_normal = rng.random(n) < 0.915
    training_required = rng.random(n) < 0.052

    professional_occ = np.isin(
        [OCCUPATIONS[i]["code18"][:2] for i in occ_idx],
        ["15", "11", "13", "17", "19", "25", "29", "27"],
    )
    coll_univ_teacher = (emp_sector == "UNIVERSITY") & np.isin(
        [OCCUPATIONS[i]["code18"][:2] for i in occ_idx], ["25"]
    )

    # Layoffs spike in the pandemic years — a genuine audit trigger.
    layoff_base = np.where(np.isin(fy, [2020, 2021]), 0.072, 0.028)
    layoff = rng.random(n) < layoff_base

    refile = rng.random(n) < 0.093

    # --- audit (latent) ------------------------------------------------------
    small_employer = emp_size < 50
    audit_logit = (
        -1.55
        + 0.55 * small_employer
        - 0.30 * (emp_size >= 5_000)
        + 0.45 * foreign_lang
        + 0.30 * alt_occ
        + 0.55 * layoff
        + 0.35 * combo_occupation
        + 0.30 * (~job_req_normal)
        + 0.25 * refile
        + rng.normal(0.0, 0.45, size=n)
    )
    audited = rng.random(n) < (1.0 / (1.0 + np.exp(-audit_logit)))

    # --- prevailing-wage determination dates --------------------------------
    # PW must be valid when the 9089 is filed; a small share have lapsed.
    pw_valid_days = rng.integers(90, 366, size=n)

    # --- outcome -------------------------------------------------------------
    fy_denial_shift = np.array([FY_DENIAL_SHIFT.get(int(y), 0.0) for y in fy])
    backlog_index = np.array([FY_BACKLOG_MEDIAN_DAYS.get(int(y), 250) for y in fy]) / 250.0

    pw_expired = rng.random(n) < 0.021

    # Base logits are tuned so the realised mix lands near the real programme's
    # shape (~88-90% certified) AFTER the positive terms below are averaged in.
    logit_certified = np.full(n, 3.45)
    logit_denied = (
        0.00
        + 2.20 * below_pw
        + 0.90 * audited
        + 0.80 * layoff
        + 0.50 * foreign_lang
        + 0.35 * alt_occ
        + 0.45 * small_employer
        - 0.25 * (emp_size >= 5_000)
        + 0.20 * refile
        + 1.00 * pw_expired
        + fy_denial_shift
        + rng.normal(0.0, 0.30, size=n)
    )
    logit_withdrawn = (
        0.20
        + 0.60 * audited
        + 0.50 * below_pw
        + 0.30 * small_employer
        + 0.35 * refile
        + 0.45 * (backlog_index - 1.0)
        + rng.normal(0.0, 0.30, size=n)
    )
    logit_expired = (
        -0.40
        + 0.40 * (backlog_index - 1.0)
        + 0.20 * small_employer
        + rng.normal(0.0, 0.30, size=n)
    )

    logits = np.column_stack([logit_certified, logit_denied, logit_withdrawn, logit_expired])
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    # Vectorised categorical sampling via the inverse CDF.
    draw = rng.random(n)[:, None]
    status_idx = (probs.cumsum(axis=1) < draw).sum(axis=1).clip(0, 3)
    case_status = np.array(CASE_STATUSES, dtype=object)[status_idx]

    # --- processing time ------------------------------------------------------
    median_days = np.array([FY_BACKLOG_MEDIAN_DAYS.get(int(y), 250) for y in fy], dtype=float)
    days = median_days * np.exp(rng.normal(0.0, 0.27, size=n))
    days += np.where(audited, rng.uniform(150.0, 430.0, size=n), 0.0)
    days *= np.where(emp_size >= 5_000, 0.94, 1.0)
    days *= np.where(small_employer, 1.07, 1.0)
    outcome_mult = np.select(
        [status_idx == 1, status_idx == 2, status_idx == 3],
        [1.12, 0.86, 1.04],
        default=1.0,
    )
    days *= outcome_mult
    processing_days = np.clip(np.round(days), 21, 1_400).astype(int)

    # --- dates ----------------------------------------------------------------
    decision = np.empty(n, dtype="datetime64[D]")
    for y, cnt in zip(years, counts):
        mask = fy == y
        start, end = _fiscal_year_bounds(int(y))
        span = int((end - start) / np.timedelta64(1, "D"))
        decision[mask] = start + rng.integers(0, span + 1, size=int(mask.sum())).astype(
            "timedelta64[D]"
        )
    received = decision - processing_days.astype("timedelta64[D]")

    pw_determ = received - rng.integers(25, 210, size=n).astype("timedelta64[D]")
    pw_expire = pw_determ + pw_valid_days.astype("timedelta64[D]")
    # Honour the pw_expired flag by pulling the expiry back before filing.
    pw_expire = np.where(
        pw_expired,
        received - rng.integers(1, 45, size=n).astype("timedelta64[D]"),
        np.maximum(pw_expire, received + np.timedelta64(1, "D")),
    )
    orig_file = np.where(
        refile,
        received - rng.integers(180, 540, size=n).astype("timedelta64[D]"),
        np.datetime64("NaT"),
    )

    # --- foreign worker -------------------------------------------------------
    citizenship = rng.choice(COUNTRIES, size=n, p=_normalise(np.array(COUNTRY_P)))
    # Birth country matches citizenship in most, but not all, cases.
    birth_country = np.where(
        rng.random(n) < 0.965, citizenship,
        rng.choice(COUNTRIES, size=n, p=_normalise(np.array(COUNTRY_P))),
    )
    class_admission = rng.choice(CLASS_OF_ADMISSION, size=n, p=_normalise(np.array(CLASS_P)))

    # The worker's education is at least the job requirement most of the time.
    fw_education = rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_P)
    job_rank = np.array([EDUCATION_RANK[e] for e in job_education])
    fw_rank = np.array([EDUCATION_RANK[e] for e in fw_education])
    upgrade = fw_rank < job_rank
    fw_education = np.where(upgrade & (rng.random(n) < 0.93), job_education, fw_education)

    edu_year = np.clip(
        (pd.DatetimeIndex(received).year.values - rng.integers(2, 22, size=n)), 1965, 2024
    )

    # --- attorney / preparer --------------------------------------------------
    self_filed = rng.random(n) < 0.118
    attorney = np.where(
        self_filed, None, rng.choice(ATTORNEY_FIRMS, size=n)
    )
    att_state = np.where(self_filed, None, rng.choice(abbrevs, size=n, p=state_p))
    att_city = np.array([
        None if s is None else STATES[s][4][int(rng.integers(len(STATES[s][4])))]
        for s in att_state
    ], dtype=object)
    preparer = rng.choice(PREPARER_TITLES, size=n, p=_normalise(np.array(PREPARER_P)))

    # --- case numbers ---------------------------------------------------------
    recv_year2 = pd.DatetimeIndex(received).year.values % 100
    recv_doy = pd.DatetimeIndex(received).dayofyear.values
    seq = rng.permutation(n)
    case_number = np.array([
        f"A-{y2:02d}{d:03d}-{s:05d}" for y2, d, s in zip(recv_year2, recv_doy, seq)
    ], dtype=object)

    # --- assemble -------------------------------------------------------------
    df = pd.DataFrame({
        "CASE_NUMBER": case_number,
        "CASE_STATUS": case_status,
        "RECEIVED_DATE": pd.to_datetime(received),
        "DECISION_DATE": pd.to_datetime(decision),
        "FISCAL_YEAR": fy.astype(int),
        "REFILE": np.where(refile, "Y", "N"),
        "ORIG_FILE_DATE": pd.to_datetime(orig_file),
        "SCHD_A_SHEEPHERDER": np.where(rng.random(n) < 0.004, "Y", "N"),
        "EMPLOYER_NAME": employers["name"][emp_idx],
        "EMPLOYER_ADDRESS_1": employers["address"][emp_idx],
        "EMPLOYER_CITY": employers["city"][emp_idx],
        "EMPLOYER_STATE": employers["state"][emp_idx],
        "EMPLOYER_POSTAL_CODE": _zip_for_state(rng, employers["state"][emp_idx]),
        "EMPLOYER_COUNTRY": "UNITED STATES OF AMERICA",
        "EMPLOYER_NUM_EMPLOYEES": emp_size.astype(float),
        "EMPLOYER_YEAR_COMMENCED_BUSINESS": employers["year_commenced"][emp_idx].astype(float),
        "NAICS_CODE": employers["naics"][emp_idx],
        "FW_OWNERSHIP_INTEREST": np.where(rng.random(n) < 0.028, "Y", "N"),
        "PW_SOC_CODE": soc_code,
        "PW_SOC_TITLE": soc_title,
        "PW_SKILL_LEVEL": skill_level,
        "PW_SOURCE": rng.choice(PW_SOURCES, size=n, p=_normalise(np.array(PW_SOURCE_P))),
        "PW_TRACK_NUM": [f"P{int(v):09d}" for v in rng.integers(1, 999_999_999, size=n)],
        "PW_DETERM_DATE": pd.to_datetime(pw_determ),
        "PW_EXPIRE_DATE": pd.to_datetime(pw_expire),
        "PW_AMOUNT_9089": pw_amount,
        "PW_UNIT_OF_PAY_9089": pw_unit,
        "WAGE_OFFER_FROM_9089": offer_from,
        "WAGE_OFFER_TO_9089": offer_to,
        "WAGE_OFFER_UNIT_OF_PAY_9089": offer_unit,
        "JOB_INFO_JOB_TITLE": job_title,
        "JOB_INFO_WORK_CITY": work_city,
        "JOB_INFO_WORK_STATE": work_state,
        "JOB_INFO_WORK_POSTAL_CODE": _zip_for_state(rng, work_state),
        "JOB_INFO_EDUCATION": job_education,
        "JOB_INFO_MAJOR": rng.choice(MAJORS, size=n),
        "JOB_INFO_TRAINING": np.where(training_required, "Y", "N"),
        "JOB_INFO_EXPERIENCE": np.where(exp_required, "Y", "N"),
        "JOB_INFO_EXPERIENCE_NUM_MONTHS": exp_months,
        "JOB_INFO_ALT_OCC": np.where(alt_occ, "Y", "N"),
        "JOB_INFO_ALT_OCC_NUM_MONTHS": alt_occ_months,
        "JOB_INFO_JOB_REQ_NORMAL": np.where(job_req_normal, "Y", "N"),
        "JOB_INFO_FOREIGN_LANG_REQ": np.where(foreign_lang, "Y", "N"),
        "JOB_INFO_COMBO_OCCUPATION": np.where(combo_occupation, "Y", "N"),
        "RECR_INFO_PROFESSIONAL_OCC": np.where(professional_occ, "Y", "N"),
        "RECR_INFO_COLL_UNIV_TEACHER": np.where(coll_univ_teacher, "Y", "N"),
        "RECR_INFO_SUNDAY_NEWSPAPER": np.where(
            professional_occ & (rng.random(n) < 0.93), "Y", "N"),
        "RI_LAYOFF_IN_PAST_SIX_MONTHS": np.where(layoff, "Y", "N"),
        "FW_INFO_BIRTH_COUNTRY": birth_country,
        "COUNTRY_OF_CITIZENSHIP": citizenship,
        "CLASS_OF_ADMISSION": class_admission,
        "FOREIGN_WORKER_INFO_EDUCATION": fw_education,
        "FW_INFO_YR_REL_EDU_COMPLETED": edu_year.astype(float),
        "FW_INFO_REQ_EXPERIENCE": np.where(exp_required & (rng.random(n) < 0.95), "Y", "N"),
        "AGENT_ATTORNEY_NAME": attorney,
        "AGENT_ATTORNEY_CITY": att_city,
        "AGENT_ATTORNEY_STATE": att_state,
        "PREPARER_INFO_TITLE": preparer,
        "_SYNTH_AUDITED": np.where(audited, "Y", "N"),
        "_SYNTH_PROCESSING_DAYS": processing_days,
        "_SYNTH_SECTOR": emp_sector,
    })

    df = _apply_missingness(df, rng)
    if config.messy:
        df = _apply_messiness(df, rng)

    # Dates become strings last so the messy format switch can see FISCAL_YEAR.
    for col in ["RECEIVED_DATE", "DECISION_DATE", "ORIG_FILE_DATE",
                "PW_DETERM_DATE", "PW_EXPIRE_DATE"]:
        df[col] = _format_dates(df[col], df["FISCAL_YEAR"].to_numpy(), config.messy)

    keep = list(COLUMN_ORDER)
    if config.include_ground_truth:
        keep += GROUND_TRUTH_COLUMNS
    df = df[keep]

    # Shuffle so rows are not ordered by fiscal year, as in a real merged extract.
    return df.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)


def _apply_missingness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Blank fields the way the real files do — conditionally, not uniformly."""
    n = len(df)

    def blank(col: str, rate: float) -> None:
        df.loc[rng.random(n) < rate, col] = np.nan

    blank("EMPLOYER_NUM_EMPLOYEES", 0.082)
    blank("EMPLOYER_YEAR_COMMENCED_BUSINESS", 0.115)
    blank("JOB_INFO_MAJOR", 0.104)
    blank("NAICS_CODE", 0.026)
    blank("EMPLOYER_POSTAL_CODE", 0.008)
    blank("JOB_INFO_WORK_POSTAL_CODE", 0.011)
    blank("PW_TRACK_NUM", 0.017)
    blank("FW_INFO_YR_REL_EDU_COMPLETED", 0.048)
    blank("PW_SKILL_LEVEL", 0.031)
    blank("EMPLOYER_ADDRESS_1", 0.004)
    return df


def _apply_messiness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the specific inconsistencies the harmonisation layer must handle."""
    n = len(df)

    df["EMPLOYER_NAME"] = _apply_name_variants(rng, df["EMPLOYER_NAME"].to_numpy())

    # CASE_STATUS casing drifts across extracts.
    upper = rng.random(n) < 0.07
    df.loc[upper, "CASE_STATUS"] = df.loc[upper, "CASE_STATUS"].str.upper()
    spaced = (rng.random(n) < 0.04) & df["CASE_STATUS"].str.contains("-", na=False)
    df.loc[spaced, "CASE_STATUS"] = df.loc[spaced, "CASE_STATUS"].str.replace(
        "-", " ", regex=False
    )

    # Worksite state occasionally spelled out.
    full = rng.random(n) < 0.035
    df.loc[full, "JOB_INFO_WORK_STATE"] = df.loc[full, "JOB_INFO_WORK_STATE"].map(
        lambda s: STATES[s][0] if s in STATES else s
    )

    # Stray whitespace in free-text fields.
    pad = rng.random(n) < 0.025
    df.loc[pad, "JOB_INFO_JOB_TITLE"] = " " + df.loc[pad, "JOB_INFO_JOB_TITLE"].astype(str)
    pad2 = rng.random(n) < 0.02
    df.loc[pad2, "JOB_INFO_WORK_CITY"] = df.loc[pad2, "JOB_INFO_WORK_CITY"].astype(str) + " "
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def preview(df: pd.DataFrame, source: Path | None = None) -> None:
    """Print shape, columns, dtypes, head, and sanity distributions."""
    sep = "=" * 78
    print(sep)
    # ASCII only: the Windows console default codepage mangles em dashes.
    print("SYNTHETIC PERM DISCLOSURE SAMPLE - NOT REAL DATA")
    if source is not None:
        print(f"file    : {source}")
    print(f"shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(sep)

    print("\nCOLUMNS")
    for i, col in enumerate(df.columns, 1):
        nulls = df[col].isna().sum()
        pct = 100.0 * nulls / len(df)
        print(f"  {i:>2}. {col:<36} {str(df[col].dtype):<10} nulls={nulls:>6,} ({pct:4.1f}%)")

    key = ["CASE_NUMBER", "CASE_STATUS", "RECEIVED_DATE", "DECISION_DATE", "FISCAL_YEAR",
           "EMPLOYER_NAME", "EMPLOYER_NUM_EMPLOYEES", "PW_SOC_CODE", "JOB_INFO_JOB_TITLE",
           "JOB_INFO_WORK_STATE", "PW_AMOUNT_9089", "WAGE_OFFER_FROM_9089",
           "WAGE_OFFER_UNIT_OF_PAY_9089"]
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\nHEAD (key columns)")
        print(df[key].head(10).to_string(index=False))

        print("\nHEAD (first row, all fields)")
        print(df.head(1).T.to_string(header=False))

    print("\nCASE_STATUS x FISCAL_YEAR (row %)")
    status = df["CASE_STATUS"].astype(str).str.upper().str.replace(" ", "-", regex=False)
    xt = pd.crosstab(df["FISCAL_YEAR"], status, normalize="index") * 100
    print(xt.round(1).to_string())

    print("\nPROCESSING DAYS by FISCAL_YEAR (derived from the two date columns)")
    recv = pd.to_datetime(df["RECEIVED_DATE"], format="mixed", errors="coerce")
    dec = pd.to_datetime(df["DECISION_DATE"], format="mixed", errors="coerce")
    dur = (dec - recv).dt.days
    print(dur.groupby(df["FISCAL_YEAR"]).describe()[
        ["count", "mean", "std", "min", "50%", "max"]
    ].round(1).to_string())

    print("\nMOST COMMON SOC CODE per FISCAL_YEAR (note the 2010 -> 2018 switch at FY2020)")
    print(df.groupby("FISCAL_YEAR")["PW_SOC_CODE"].agg(
        lambda s: s.value_counts().index[0]).to_string())
    print(sep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic PERM disclosure dataset (NOT real data).",
    )
    parser.add_argument("--rows", type=int, default=50_000, help="total rows (default 50000)")
    parser.add_argument("--employers", type=int, default=1_400,
                        help="size of the synthetic employer pool (default 1400)")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS),
                        help="decision fiscal years to cover (default 2018..2024)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output CSV (default {DEFAULT_OUTPUT})")
    parser.add_argument("--no-messy", dest="messy", action="store_false",
                        help="emit clean, already-harmonised values")
    parser.add_argument("--include-ground-truth", action="store_true",
                        help="also write _SYNTH_* latent columns (LEAKS the answer)")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="skip the printed preview")
    args = parser.parse_args(argv)

    config = GeneratorConfig(
        n_rows=args.rows,
        n_employers=args.employers,
        years=tuple(args.years),
        seed=args.seed,
        messy=args.messy,
        include_ground_truth=args.include_ground_truth,
        output_path=args.out,
    )

    df = generate(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_path, index=False)

    if args.preview:
        preview(df, source=config.output_path)

    size_mb = config.output_path.stat().st_size / 1_048_576
    print(f"\nwrote {len(df):,} rows to {config.output_path} ({size_mb:.1f} MB)")
    print(f"generator v{GENERATOR_VERSION}  seed={config.seed}  messy={config.messy}  "
          f"ground_truth={config.include_ground_truth}")
    print("Reminder: synthetic data. No metric from this file is a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
