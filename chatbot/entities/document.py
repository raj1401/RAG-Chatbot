from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from pydantic import Field


@dataclass
class Document:
    """
    Class for storing the document information.
    """
    page_content: str
    metadata: dict = Field(default_factory=dict)
    type: Literal["Document"] = "Document"