"""Static master data for the fictional SaaS company: entities, cost centers, chart of accounts, vendors."""

from __future__ import annotations

import pandas as pd

ENTITIES = pd.DataFrame(
    [
        ("US01", "EZ Tech, Inc.", "USD", "United States"),
        ("UK01", "EZ Tech Ltd.", "GBP", "United Kingdom"),
        ("DE01", "EZ Tech GmbH", "EUR", "Germany"),
    ],
    columns=["entity", "entity_name", "currency", "country"],
)

# Share of the customer base (and subscription revenue) booked in each entity.
ENTITY_REVENUE_SHARE = {"US01": 0.70, "UK01": 0.20, "DE01": 0.10}

# department -> functional P&L grouping used on the executive summary
DEPARTMENTS = {
    "Sales": "Sales & Marketing",
    "Marketing": "Sales & Marketing",
    "Customer Success": "Sales & Marketing",
    "Engineering": "Research & Development",
    "Product": "Research & Development",
    "Support": "Cost of Revenue",
    "Professional Services": "Cost of Revenue",
    "Cloud Operations": "Cost of Revenue",
    "G&A": "General & Administrative",
    "Company": "Revenue",
}

# cost_center, name, department, entity, budget owner (fictional), starting headcount (Jan-2021),
# fully loaded annual salary in USD at Jan-2021, headcount elasticity to ARR growth
COST_CENTERS = pd.DataFrame(
    [
        ("CC-000", "Corporate Revenue", "Company", "US01", "Dana Whitaker (CFO)", 0, 0, 0.0),
        ("CC-110", "Sales - North America", "Sales", "US01", "Marcus Hale", 17, 105_000, 0.75),
        ("CC-120", "Sales - EMEA", "Sales", "UK01", "Priya Natarajan", 6, 95_000, 0.80),
        ("CC-210", "Marketing", "Marketing", "US01", "Elena Brooks", 10, 118_000, 0.60),
        ("CC-310", "Customer Success", "Customer Success", "US01", "Tom Okafor", 9, 92_000, 0.70),
        ("CC-410", "Engineering - Platform", "Engineering", "US01", "Wei Zhang", 27, 168_000, 0.55),
        ("CC-420", "Engineering - Data", "Engineering", "DE01", "Lukas Fischer", 7, 128_000, 0.65),
        ("CC-510", "Product Management", "Product", "US01", "Sofia Martins", 5, 160_000, 0.55),
        ("CC-610", "Finance", "G&A", "US01", "Dana Whitaker (CFO)", 5, 135_000, 0.40),
        ("CC-620", "People & Talent", "G&A", "US01", "Grace Kim", 4, 115_000, 0.50),
        ("CC-630", "Legal", "G&A", "US01", "Jonah Reyes", 2, 190_000, 0.35),
        ("CC-640", "IT", "G&A", "US01", "Sam Patel", 3, 120_000, 0.45),
        ("CC-710", "Support", "Support", "UK01", "Aoife Byrne", 8, 62_000, 0.70),
        ("CC-720", "Professional Services", "Professional Services", "US01", "Nina Alvarez", 7, 112_000, 0.60),
        ("CC-810", "Cloud Operations", "Cloud Operations", "US01", "Ravi Shah", 4, 158_000, 0.50),
    ],
    columns=[
        "cost_center",
        "cost_center_name",
        "department",
        "entity",
        "budget_owner",
        "hc_start",
        "salary_start_usd",
        "hc_elasticity",
    ],
)

# account, name, type, financial-statement line, normal balance
CHART_OF_ACCOUNTS = pd.DataFrame(
    [
        ("1000", "Cash - Operating", "Asset", None, "Debit"),
        ("1200", "Accounts Receivable", "Asset", None, "Debit"),
        ("1590", "Accumulated Depreciation", "Asset", None, "Credit"),
        ("2000", "Accounts Payable", "Liability", None, "Credit"),
        ("2100", "Accrued Liabilities", "Liability", None, "Credit"),
        ("2150", "Accrued Payroll & Bonus", "Liability", None, "Credit"),
        ("2160", "Accrued Commissions", "Liability", None, "Credit"),
        ("2400", "Deferred Revenue", "Liability", None, "Credit"),
        ("4000", "Subscription Revenue", "Revenue", "Subscription Revenue", "Credit"),
        ("4100", "Professional Services Revenue", "Revenue", "Services Revenue", "Credit"),
        ("5000", "Cloud Hosting", "Cost of Revenue", "Hosting", "Debit"),
        ("5010", "Monitoring & Hosting Tools", "Cost of Revenue", "Hosting", "Debit"),
        ("6000", "Salaries & Wages", "Operating Expense", "Personnel", "Debit"),
        ("6010", "Bonus", "Operating Expense", "Personnel", "Debit"),
        ("6020", "Payroll Taxes & Benefits", "Operating Expense", "Personnel", "Debit"),
        ("6100", "Sales Commissions", "Operating Expense", "Commissions", "Debit"),
        ("6200", "Contractors & Consultants", "Operating Expense", "Contractors", "Debit"),
        ("6300", "Software & Subscriptions", "Operating Expense", "Software & Tools", "Debit"),
        ("6400", "Marketing Programs", "Operating Expense", "Marketing Programs", "Debit"),
        ("6410", "Events & Trade Shows", "Operating Expense", "Marketing Programs", "Debit"),
        ("6500", "Travel & Entertainment", "Operating Expense", "Travel & Entertainment", "Debit"),
        ("6600", "Rent & Facilities", "Operating Expense", "Facilities", "Debit"),
        ("6700", "Professional Fees", "Operating Expense", "Professional Fees", "Debit"),
        ("6800", "Depreciation & Amortization", "Operating Expense", "Depreciation", "Debit"),
        ("6900", "Recruiting", "Operating Expense", "Recruiting", "Debit"),
    ],
    columns=["account", "account_name", "account_type", "fs_line", "normal_balance"],
)

VENDORS = pd.DataFrame(
    [
        ("V1001", "Stratosphere Cloud Services", "Hosting"),
        ("V1002", "Nimbus Compute LLC", "Hosting"),
        ("V1003", "Observa Monitoring", "Hosting"),
        ("V2001", "DevForce Consulting", "Contractors"),
        ("V2002", "CodeBridge Partners", "Contractors"),
        ("V2003", "Talent Stack LLC", "Contractors"),
        ("V2004", "Ledgerly Advisory", "Contractors"),
        ("V2005", "Quantum Staffing Group", "Contractors"),
        ("V3001", "PipelineCRM Inc.", "Software"),
        ("V3002", "ChatWorks", "Software"),
        ("V3003", "DocuFlow", "Software"),
        ("V3004", "PeopleSuite HRIS", "Software"),
        ("V3005", "CodeHub", "Software"),
        ("V3006", "TicketDesk", "Software"),
        ("V3007", "BI Studio", "Software"),
        ("V4001", "AdReach Media", "Marketing"),
        ("V4002", "SearchSpark Ads", "Marketing"),
        ("V4003", "EventHorizon Productions", "Marketing"),
        ("V4004", "ContentLab Agency", "Marketing"),
        ("V4005", "SaaS Summit Expo", "Marketing"),
        ("V5001", "Corporate Card Program", "T&E"),
        ("V6001", "Harborview Properties LLC", "Facilities"),
        ("V6002", "CleanSweep Facilities", "Facilities"),
        ("V7001", "Whitfield & Crane LLP", "Professional Fees"),
        ("V7002", "Morrow Legal Group", "Professional Fees"),
        ("V7003", "TaxPoint Advisors", "Professional Fees"),
        ("V8001", "TalentScout Recruiting", "Recruiting"),
        ("V8002", "JobBoard Pro", "Recruiting"),
    ],
    columns=["vendor_id", "vendor_name", "vendor_category"],
)

# Which (fs_line, cost_center) combinations carry spend. Personnel is generated for every
# cost center with headcount; the rest are listed explicitly.
LINE_COST_CENTERS = {
    "Subscription Revenue": ["CC-000"],
    "Services Revenue": ["CC-000"],
    "Hosting": ["CC-810"],
    "Commissions": ["CC-110", "CC-120"],
    "Contractors": ["CC-410", "CC-420", "CC-720", "CC-610"],
    "Software & Tools": ["CC-640", "CC-410"],
    "Marketing Programs": ["CC-210"],
    "Travel & Entertainment": ["CC-110", "CC-120", "CC-310", "CC-210", "CC-610"],
    "Facilities": ["CC-610"],
    "Professional Fees": ["CC-630", "CC-610"],
    "Depreciation": ["CC-610"],
    "Recruiting": ["CC-620"],
}


def cost_center_lookup() -> dict[str, dict]:
    return COST_CENTERS.set_index("cost_center").to_dict("index")


def master_tables() -> dict[str, pd.DataFrame]:
    """Master-data extracts written alongside the GL, as an ERP would export them."""
    coa = CHART_OF_ACCOUNTS.copy()
    cc = COST_CENTERS[["cost_center", "cost_center_name", "department", "entity", "budget_owner"]].copy()
    cc["function"] = cc["department"].map(DEPARTMENTS)
    return {
        "entities": ENTITIES.copy(),
        "chart_of_accounts": coa,
        "cost_centers": cc,
        "vendors": VENDORS.copy(),
    }
