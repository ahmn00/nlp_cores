from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EventDefinition:
    event_type: str
    description: str
    properties_schema: dict[str, Any]
    persona_affinity: list[str]  # archetype slugs that prefer this event


BLOOMBERG_EVENTS: list[EventDefinition] = [
    EventDefinition(
        event_type="session_start",
        description="User opens a Bloomberg Terminal session",
        properties_schema={
            "login_method": ["password", "biometric", "sso"],
        },
        persona_affinity=["all"],
    ),
    EventDefinition(
        event_type="session_end",
        description="User closes or times out of a Bloomberg Terminal session",
        properties_schema={
            "duration_seconds": "int",
            "logout_type": ["manual", "timeout", "idle"],
        },
        persona_affinity=["all"],
    ),
    EventDefinition(
        event_type="screen_view",
        description="User navigates to a Bloomberg function screen",
        properties_schema={
            "function_code": ["TAX", "BTAX", "NI", "LAW", "BLAW", "CRED", "CF", "MA", "BRIEF", "COURT", "REGN"],
            "previous_screen": "string",
        },
        persona_affinity=["all"],
    ),
    EventDefinition(
        event_type="search_query",
        description="User submits a search query in Bloomberg",
        properties_schema={
            "query_text": "string",
            "search_scope": ["full_text", "news", "document", "case_law", "regulation"],
            "result_count": "int",
        },
        persona_affinity=["senior_tax_partner", "tax_research_analyst", "tax_litigator", "junior_tax_associate"],
    ),
    EventDefinition(
        event_type="document_view",
        description="User opens a tax law, ruling, or news article",
        properties_schema={
            "doc_id": "string",
            "doc_type": ["IRS_ruling", "tax_regulation", "case_law", "treaty", "guidance", "commentary", "news_article"],
            "doc_title": "string",
            "jurisdiction": ["US_federal", "US_state", "EU", "UK", "OECD", "international"],
        },
        persona_affinity=["senior_tax_partner", "tax_research_analyst", "junior_tax_associate", "tax_litigator", "estate_planning_attorney"],
    ),
    EventDefinition(
        event_type="document_download",
        description="User exports or saves a document",
        properties_schema={
            "doc_id": "string",
            "doc_type": ["IRS_ruling", "tax_regulation", "case_law", "treaty", "guidance", "commentary"],
            "export_format": ["PDF", "DOCX", "CSV", "Excel"],
        },
        persona_affinity=["senior_tax_partner", "junior_tax_associate", "tax_research_analyst", "compliance_officer"],
    ),
    EventDefinition(
        event_type="news_view",
        description="User views a Bloomberg news item",
        properties_schema={
            "story_id": "string",
            "headline": "string",
            "topic_tags": "list[string]",
            "source": ["Bloomberg Tax", "Reuters", "Bloomberg Law", "Tax Notes", "BNA"],
        },
        persona_affinity=["senior_tax_partner", "inhouse_tax_counsel", "tax_technology_manager", "compliance_officer"],
    ),
    EventDefinition(
        event_type="alert_create",
        description="User sets up a keyword or company alert",
        properties_schema={
            "alert_type": ["keyword", "company", "regulation", "case"],
            "alert_query": "string",
            "frequency": ["real_time", "daily", "weekly"],
        },
        persona_affinity=["senior_tax_partner", "inhouse_tax_counsel", "compliance_officer", "tax_research_analyst"],
    ),
    EventDefinition(
        event_type="alert_delete",
        description="User removes an existing alert",
        properties_schema={
            "alert_id": "string",
            "alert_type": ["keyword", "company", "regulation", "case"],
        },
        persona_affinity=["senior_tax_partner", "inhouse_tax_counsel", "compliance_officer"],
    ),
    EventDefinition(
        event_type="data_export",
        description="User exports a dataset such as a tax rates table",
        properties_schema={
            "dataset_type": ["tax_rates", "transfer_pricing", "treaty_table", "case_list", "regulation_index"],
            "row_count": "int",
            "export_format": ["CSV", "Excel", "JSON"],
            "date_range": "string",
        },
        persona_affinity=["transfer_pricing_specialist", "tax_technology_manager", "tax_research_analyst", "compliance_officer"],
    ),
    EventDefinition(
        event_type="function_run",
        description="User runs a Bloomberg terminal function",
        properties_schema={
            "function_code": ["TAX", "BTAX", "NI", "LAW", "BLAW", "CRED", "CF", "MA", "BRIEF", "COURT", "REGN", "TRA", "TRAX"],
            "input_params": "dict",
        },
        persona_affinity=["ma_tax_specialist", "transfer_pricing_specialist", "tax_technology_manager", "inhouse_tax_counsel"],
    ),
    EventDefinition(
        event_type="annotation_add",
        description="User highlights or annotates a document",
        properties_schema={
            "doc_id": "string",
            "annotation_type": ["highlight", "note", "bookmark", "tag"],
            "note_text": "string",
        },
        persona_affinity=["senior_tax_partner", "junior_tax_associate", "tax_litigator", "estate_planning_attorney"],
    ),
    EventDefinition(
        event_type="contact_lookup",
        description="User looks up a professional contact",
        properties_schema={
            "lookup_type": ["attorney", "regulator", "tax_authority", "expert_witness"],
            "query": "string",
        },
        persona_affinity=["senior_tax_partner", "tax_litigator", "estate_planning_attorney", "ma_tax_specialist"],
    ),
]

EVENT_MAP: dict[str, EventDefinition] = {e.event_type: e for e in BLOOMBERG_EVENTS}
EVENT_TYPES: list[str] = [e.event_type for e in BLOOMBERG_EVENTS]
