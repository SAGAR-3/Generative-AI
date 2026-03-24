"""
Pydantic data models used across the application.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class TestType(str, Enum):
    FUNCTIONAL   = "functional"
    EDGE         = "edge"
    NEGATIVE     = "negative"
    PERFORMANCE  = "performance"
    SECURITY     = "security"
    ACCESSIBILITY= "accessibility"
    REGRESSION   = "regression"
    UAT          = "uat"


class Priority(str, Enum):
    P0 = "P0 - Critical"
    P1 = "P1 - High"
    P2 = "P2 - Medium"
    P3 = "P3 - Low"


class TestStep(BaseModel):
    step_number: int
    action:      str
    expected:    str


class TestCase(BaseModel):
    id:                   str          = Field(default_factory=lambda: f"TC-{str(uuid.uuid4())[:8].upper()}")
    title:                str
    test_type:            TestType
    priority:             Priority
    preconditions:        list[str]    = []
    steps:                list[TestStep]
    expected_result:      str
    linked_requirement:   str          = ""
    tags:                 list[str]    = []
    status:               str          = "draft"   # draft | approved | rejected


class BRDDocument(BaseModel):
    id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename:  str
    content:   str
    chunks:    list[str] = []


class GenerateRequest(BaseModel):
    document_id:  str
    test_types:   list[TestType] = list(TestType)
    max_cases:    int            = 50
    model:        str            = "claude-sonnet-4-20250514"


class GenerateResponse(BaseModel):
    document_id:   str
    test_cases:    list[TestCase]
    total:         int
    coverage_gaps: list[str]     = []


class ReviewAction(BaseModel):
    test_case_id: str
    action:       str   # approve | reject | edit
    edited_case:  Optional[TestCase] = None
    feedback:     Optional[str]      = None
