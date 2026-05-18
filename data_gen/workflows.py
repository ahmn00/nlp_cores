"""
Named workflows for Bloomberg tax/legal users.

Each workflow is an ordered sequence of event steps that simulates a coherent
real-world task (e.g. "case_law_research", "ma_due_diligence"). Workflows are
used as ground-truth sequence labels for LSTM/transformer encoding tasks.

Sessions share a context dict (doc_id, doc_type, doc_title, query, function_code)
so that view -> annotate -> download events reference the same document.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkflowStep:
    event_type: str
    repeat_min: int = 1
    repeat_max: int = 1


@dataclass
class WorkflowDefinition:
    name: str
    label: str
    persona_affinity: list[str]
    steps: list[WorkflowStep]
    primary_doc_type: Optional[str] = None
    primary_function: Optional[str] = None
    length_class: str = "medium"  # "short" | "medium" | "long"


WORKFLOWS: list[WorkflowDefinition] = [
    # === Research / Document Analysis ===
    WorkflowDefinition(
        name="case_law_research",
        label="Tax Case Law Research",
        persona_affinity=["senior_tax_partner", "tax_litigator", "junior_tax_associate", "tax_research_analyst"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=4),
            WorkflowStep("annotation_add", repeat_min=1, repeat_max=2),
            WorkflowStep("document_download"),
        ],
        primary_doc_type="case_law",
        primary_function="LAW",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="irs_ruling_lookup",
        label="IRS Ruling Lookup",
        persona_affinity=["senior_tax_partner", "junior_tax_associate", "compliance_officer", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=3),
            WorkflowStep("annotation_add"),
            WorkflowStep("document_download"),
        ],
        primary_doc_type="IRS_ruling",
        primary_function="TAX",
        length_class="short",
    ),
    WorkflowDefinition(
        name="regulation_review",
        label="Tax Regulation Review",
        persona_affinity=["compliance_officer", "inhouse_tax_counsel", "transfer_pricing_specialist", "senior_tax_partner"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=3),
            WorkflowStep("annotation_add", repeat_min=1, repeat_max=3),
        ],
        primary_doc_type="tax_regulation",
        primary_function="REGN",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="treaty_analysis",
        label="Tax Treaty Analysis",
        persona_affinity=["transfer_pricing_specialist", "senior_tax_partner", "ma_tax_specialist", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("function_run"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=3),
            WorkflowStep("document_download"),
        ],
        primary_doc_type="treaty",
        primary_function="TAX",
        length_class="medium",
    ),

    # === M&A / Transactions ===
    WorkflowDefinition(
        name="ma_due_diligence",
        label="M&A Tax Due Diligence",
        persona_affinity=["ma_tax_specialist", "senior_tax_partner", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("search_query", repeat_min=1, repeat_max=2),
            WorkflowStep("document_view", repeat_min=3, repeat_max=5),
            WorkflowStep("annotation_add", repeat_min=2, repeat_max=3),
            WorkflowStep("document_download", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="commentary",
        primary_function="MA",
        length_class="long",
    ),
    WorkflowDefinition(
        name="deal_tax_structuring",
        label="Deal Tax Structuring Analysis",
        persona_affinity=["ma_tax_specialist", "senior_tax_partner"],
        steps=[
            WorkflowStep("function_run"),
            WorkflowStep("function_run"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=3),
            WorkflowStep("data_export"),
            WorkflowStep("document_download"),
        ],
        primary_doc_type="tax_regulation",
        primary_function="MA",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="tax_modeling",
        label="Tax Modeling and Data Analysis",
        persona_affinity=["ma_tax_specialist", "tax_technology_manager", "transfer_pricing_specialist"],
        steps=[
            WorkflowStep("function_run"),
            WorkflowStep("data_export"),
            WorkflowStep("function_run"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="guidance",
        primary_function="BTAX",
        length_class="medium",
    ),

    # === Transfer Pricing ===
    WorkflowDefinition(
        name="tp_comparables_search",
        label="Transfer Pricing Comparables Search",
        persona_affinity=["transfer_pricing_specialist", "ma_tax_specialist"],
        steps=[
            WorkflowStep("function_run"),
            WorkflowStep("data_export"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="treaty",
        primary_function="TRAX",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="beps_compliance_review",
        label="BEPS / Pillar Two Compliance Review",
        persona_affinity=["transfer_pricing_specialist", "compliance_officer", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=4),
            WorkflowStep("alert_create"),
        ],
        primary_doc_type="guidance",
        primary_function="REGN",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="tp_documentation",
        label="Transfer Pricing Documentation",
        persona_affinity=["transfer_pricing_specialist", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=4),
            WorkflowStep("annotation_add", repeat_min=2, repeat_max=3),
            WorkflowStep("document_download", repeat_min=1, repeat_max=2),
            WorkflowStep("data_export"),
        ],
        primary_doc_type="tax_regulation",
        primary_function="TRA",
        length_class="long",
    ),

    # === Compliance / Monitoring ===
    WorkflowDefinition(
        name="regulatory_news_scan",
        label="Regulatory News Monitoring",
        persona_affinity=["compliance_officer", "inhouse_tax_counsel", "tax_research_analyst", "senior_tax_partner"],
        steps=[
            WorkflowStep("news_view", repeat_min=3, repeat_max=6),
            WorkflowStep("alert_create", repeat_min=1, repeat_max=2),
            WorkflowStep("search_query"),
            WorkflowStep("document_view"),
        ],
        primary_doc_type="news_article",
        primary_function="NI",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="audit_risk_assessment",
        label="Tax Audit Risk Assessment",
        persona_affinity=["compliance_officer", "senior_tax_partner", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=3, repeat_max=4),
            WorkflowStep("document_download", repeat_min=1, repeat_max=2),
            WorkflowStep("data_export"),
        ],
        primary_doc_type="IRS_ruling",
        primary_function="TAX",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="filing_deadline_check",
        label="Filing Deadline and Compliance Check",
        persona_affinity=["compliance_officer", "inhouse_tax_counsel", "tax_technology_manager"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=2),
            WorkflowStep("alert_create"),
        ],
        primary_doc_type="guidance",
        primary_function="TAX",
        length_class="short",
    ),

    # === Litigation ===
    WorkflowDefinition(
        name="irs_dispute_research",
        label="IRS Dispute and Controversy Research",
        persona_affinity=["tax_litigator", "senior_tax_partner"],
        steps=[
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=3),
            WorkflowStep("contact_lookup"),
            WorkflowStep("annotation_add", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="IRS_ruling",
        primary_function="COURT",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="brief_preparation",
        label="Legal Brief Research and Preparation",
        persona_affinity=["tax_litigator", "junior_tax_associate"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("search_query", repeat_min=1, repeat_max=2),
            WorkflowStep("document_view", repeat_min=3, repeat_max=5),
            WorkflowStep("annotation_add", repeat_min=2, repeat_max=4),
            WorkflowStep("document_download", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="case_law",
        primary_function="BRIEF",
        length_class="long",
    ),
    WorkflowDefinition(
        name="expert_witness_prep",
        label="Expert Witness Research",
        persona_affinity=["tax_litigator", "senior_tax_partner"],
        steps=[
            WorkflowStep("contact_lookup", repeat_min=1, repeat_max=3),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="case_law",
        primary_function="COURT",
        length_class="short",
    ),

    # === Estate Planning ===
    WorkflowDefinition(
        name="estate_plan_research",
        label="Estate and Gift Tax Planning Research",
        persona_affinity=["estate_planning_attorney", "senior_tax_partner"],
        steps=[
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=3),
            WorkflowStep("annotation_add", repeat_min=1, repeat_max=2),
            WorkflowStep("document_download"),
        ],
        primary_doc_type="guidance",
        primary_function="TAX",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="trust_structure_review",
        label="Trust and Estate Structure Review",
        persona_affinity=["estate_planning_attorney"],
        steps=[
            WorkflowStep("function_run"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=2, repeat_max=4),
            WorkflowStep("annotation_add", repeat_min=1, repeat_max=3),
        ],
        primary_doc_type="tax_regulation",
        primary_function="TAX",
        length_class="medium",
    ),

    # === News / Light usage ===
    WorkflowDefinition(
        name="morning_news_briefing",
        label="Morning News and Market Briefing",
        persona_affinity=["senior_tax_partner", "inhouse_tax_counsel", "compliance_officer", "tax_research_analyst", "ma_tax_specialist"],
        steps=[
            WorkflowStep("news_view", repeat_min=4, repeat_max=8),
            WorkflowStep("alert_create"),
        ],
        primary_doc_type="news_article",
        primary_function="NI",
        length_class="short",
    ),
    WorkflowDefinition(
        name="legislative_tracking",
        label="Legislative and Policy Tracking",
        persona_affinity=["tax_research_analyst", "compliance_officer", "inhouse_tax_counsel"],
        steps=[
            WorkflowStep("screen_view"),
            WorkflowStep("news_view", repeat_min=2, repeat_max=4),
            WorkflowStep("alert_create"),
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="news_article",
        primary_function="NI",
        length_class="medium",
    ),
    WorkflowDefinition(
        name="quick_reference_lookup",
        label="Quick Reference Lookup",
        persona_affinity=["all"],
        steps=[
            WorkflowStep("search_query"),
            WorkflowStep("document_view", repeat_min=1, repeat_max=2),
        ],
        primary_doc_type="commentary",
        primary_function="LAW",
        length_class="short",
    ),
    WorkflowDefinition(
        name="tax_data_export",
        label="Tax Data and Rates Export",
        persona_affinity=["tax_technology_manager", "tax_research_analyst", "transfer_pricing_specialist", "compliance_officer"],
        steps=[
            WorkflowStep("function_run"),
            WorkflowStep("data_export", repeat_min=1, repeat_max=2),
            WorkflowStep("function_run"),
        ],
        primary_doc_type="guidance",
        primary_function="BTAX",
        length_class="short",
    ),
]

WORKFLOW_MAP: dict[str, WorkflowDefinition] = {w.name: w for w in WORKFLOWS}
WORKFLOW_NAMES: list[str] = [w.name for w in WORKFLOWS]


def get_workflows_for_persona(archetype: str) -> list[WorkflowDefinition]:
    """Return workflows relevant to a given persona archetype."""
    return [
        w for w in WORKFLOWS
        if archetype in w.persona_affinity or "all" in w.persona_affinity
    ]
