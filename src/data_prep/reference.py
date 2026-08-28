"""Reference lookup tables for PERM data preparation.

Pure data, no logic. Kept separate so the cleaning rules stay readable and the
crosswalks can be reviewed (and corrected) on their own.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
STATE_NAME_TO_ABBREV = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    # Territories and common alternates seen in disclosure files.
    "PUERTO RICO": "PR", "GUAM": "GU", "VIRGIN ISLANDS": "VI",
    "U.S. VIRGIN ISLANDS": "VI", "AMERICAN SAMOA": "AS",
    "NORTHERN MARIANA ISLANDS": "MP", "WASHINGTON DC": "DC", "WASHINGTON D.C.": "DC",
    "D.C.": "DC",
}

VALID_STATE_ABBREVS = set(STATE_NAME_TO_ABBREV.values())

# U.S. Census Bureau regions — the grouping used for the fairness audit's
# geographic slice.
STATE_TO_REGION = {
    **{s: "Northeast" for s in
       ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"]},
    **{s: "Midwest" for s in
       ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"]},
    **{s: "South" for s in
       ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS",
        "TN", "AR", "LA", "OK", "TX"]},
    **{s: "West" for s in
       ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"]},
    **{s: "Territory" for s in ["PR", "GU", "VI", "AS", "MP"]},
}


# ---------------------------------------------------------------------------
# Wage units
# ---------------------------------------------------------------------------
# Multiplier that converts a wage stated in the unit to an annual figure.
# 2,080 = 40 h/week x 52 weeks, the OFLC convention.
WAGE_UNIT_TO_ANNUAL = {
    "YEAR": 1.0, "YR": 1.0, "Y": 1.0, "ANNUAL": 1.0, "ANNUALLY": 1.0, "A": 1.0,
    "HOUR": 2080.0, "HR": 2080.0, "H": 2080.0, "HOURLY": 2080.0,
    "MONTH": 12.0, "MTH": 12.0, "MO": 12.0, "M": 12.0, "MONTHLY": 12.0,
    "WEEK": 52.0, "WK": 52.0, "W": 52.0, "WEEKLY": 52.0,
    "BI-WEEKLY": 26.0, "BIWEEKLY": 26.0, "BI WEEKLY": 26.0, "BW": 26.0, "BI": 26.0,
}

# Annualised wages outside this band are treated as data errors, not signal.
# The floor sits below a full-time federal minimum wage year (~$15k); the
# ceiling is well above any plausible PERM filing.
MIN_PLAUSIBLE_ANNUAL_WAGE = 10_000.0
MAX_PLAUSIBLE_ANNUAL_WAGE = 5_000_000.0


# ---------------------------------------------------------------------------
# Case status
# ---------------------------------------------------------------------------
# Maps the many spellings seen across extracts onto four canonical outcomes.
# Keys are compared after uppercasing and collapsing separators to a hyphen.
CASE_STATUS_MAP = {
    "CERTIFIED": "certified",
    "CERTIFIED-EXPIRED": "certified_expired",
    "CERTIFIEDEXPIRED": "certified_expired",
    "CERTIFIED-EXPIRE": "certified_expired",
    "DENIED": "denied",
    "DENY": "denied",
    "WITHDRAWN": "withdrawn",
    "WITHDRAW": "withdrawn",
}

# Statuses that mean the case had not reached a final determination. Rows
# carrying these are dropped: they have no outcome and no true processing time.
NON_FINAL_STATUSES = {
    "PENDING", "IN PROCESS", "PENDING QUALITY CONTROL", "PENDING-QUALITY-CONTROL",
    "INCOMPLETE", "RECEIVED", "PROCESSING",
}


# ---------------------------------------------------------------------------
# SOC crosswalk: 2010 -> 2018
# ---------------------------------------------------------------------------
# The 2018 SOC revision renamed and merged a number of occupations. Disclosure
# files use 2010 codes for the earlier fiscal years and 2018 codes later, so the
# same job appears under two codes on either side of the changeover. Without
# this crosswalk a model sees two unrelated categories split on the train/test
# boundary.
#
# Caveat worth stating in the write-up: the official crosswalk is many-to-many
# in places. 15-2041 (Statisticians) is the clearest case — the 2018 revision
# carved Data Scientists (15-2051) out of work previously coded under
# Statisticians and Operations Research Analysts. Collapsing 15-2041 to 15-2051
# keeps the series continuous but is a judgement call, not a lossless mapping.
# `soc_major_group` (the first two digits) is unaffected by all of this and is
# the safer feature when the specific code is not needed.
SOC_2010_TO_2018 = {
    # Computer and mathematical occupations
    "15-1111": "15-1221",  # Computer and Information Research Scientists
    "15-1121": "15-1211",  # Computer Systems Analysts
    "15-1122": "15-1212",  # Information Security Analysts
    "15-1131": "15-1251",  # Computer Programmers
    "15-1132": "15-1252",  # Software Developers, Applications
    "15-1133": "15-1252",  # Software Developers, Systems Software
    "15-1134": "15-1254",  # Web Developers
    "15-1141": "15-1242",  # Database Administrators
    "15-1142": "15-1244",  # Network and Computer Systems Administrators
    "15-1143": "15-1241",  # Computer Network Architects
    "15-1151": "15-1232",  # Computer User Support Specialists
    "15-1152": "15-1231",  # Computer Network Support Specialists
    "15-1199": "15-1299",  # Computer Occupations, All Other
    "15-2011": "15-2011",  # Actuaries
    "15-2031": "15-2031",  # Operations Research Analysts
    "15-2041": "15-2051",  # Statisticians -> see caveat above
    # Healthcare practitioners
    "29-1062": "29-1215",  # Family and General Practitioners
    "29-1063": "29-1216",  # Internists, General
    "29-1065": "29-1221",  # Pediatricians, General
    "29-1067": "29-1248",  # Surgeons
    "29-1069": "29-1229",  # Physicians and Surgeons, All Other
    "29-1111": "29-1141",  # Registered Nurses
    # Sales
    "41-3099": "41-3091",  # Sales Representatives, Services, All Other
    # Media and design
    "27-1014": "27-1014",  # Special Effects Artists and Animators
    "27-3031": "27-3031",  # Public Relations Specialists
}

# Codes retired in 2018 with no single successor. Left as-is and flagged rather
# than force-mapped; `soc_major_group` still works for them.
SOC_2010_UNMAPPED_NOTE = (
    "codes not in the crosswalk are passed through unchanged; "
    "use soc_major_group if continuity matters"
)


# ---------------------------------------------------------------------------
# SOC display titles (2018 SOC)
# ---------------------------------------------------------------------------
# Display only. Nothing in the pipeline keys off these -- they exist so an
# interface can show "15-1252 - Software Developers" rather than a bare code.
SOC_CODE_TITLES = {
    "11-1021": "General and Operations Managers",
    "11-2021": "Marketing Managers",
    "11-3021": "Computer and Information Systems Managers",
    "11-3031": "Financial Managers",
    "13-1111": "Management Analysts",
    "13-1161": "Market Research Analysts and Marketing Specialists",
    "13-2011": "Accountants and Auditors",
    "13-2051": "Financial and Investment Analysts",
    "15-1211": "Computer Systems Analysts",
    "15-1221": "Computer and Information Research Scientists",
    "15-1231": "Computer Network Support Specialists",
    "15-1232": "Computer User Support Specialists",
    "15-1241": "Computer Network Architects",
    "15-1242": "Database Administrators",
    "15-1244": "Network and Computer Systems Administrators",
    "15-1251": "Computer Programmers",
    "15-1252": "Software Developers",
    "15-1254": "Web Developers",
    "15-1299": "Computer Occupations, All Other",
    "15-2011": "Actuaries",
    "15-2031": "Operations Research Analysts",
    "15-2051": "Data Scientists",
    "17-2051": "Civil Engineers",
    "17-2061": "Computer Hardware Engineers",
    "17-2071": "Electrical Engineers",
    "17-2112": "Industrial Engineers",
    "17-2141": "Mechanical Engineers",
    "19-1042": "Medical Scientists, Except Epidemiologists",
    "19-2031": "Chemists",
    "19-3011": "Economists",
    "25-1022": "Mathematical Science Teachers, Postsecondary",
    "25-1042": "Biological Science Teachers, Postsecondary",
    "25-1071": "Health Specialties Teachers, Postsecondary",
    "25-2021": "Elementary School Teachers, Except Special Education",
    "27-1024": "Graphic Designers",
    "29-1051": "Pharmacists",
    "29-1123": "Physical Therapists",
    "29-1141": "Registered Nurses",
    "29-1215": "Family Medicine Physicians",
    "29-1229": "Physicians, All Other",
    "35-1011": "Chefs and Head Cooks",
    "41-3091": "Sales Representatives of Services, All Other",
    "51-1011": "First-Line Supervisors of Production and Operating Workers",
}

SOC_MAJOR_GROUP_TITLES = {
    "11": "Management",
    "13": "Business and Financial Operations",
    "15": "Computer and Mathematical",
    "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science",
    "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media",
    "29": "Healthcare Practitioners and Technical",
    "35": "Food Preparation and Serving",
    "41": "Sales and Related",
    "43": "Office and Administrative Support",
    "51": "Production",
    "53": "Transportation and Material Moving",
}
