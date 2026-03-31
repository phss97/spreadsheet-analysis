from __future__ import annotations

import os
import re
import tempfile
from io import StringIO
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from crewai import LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class SpreadsheetAnalysisInput(BaseModel):
    """Input schema for the Spreadsheet Analysis tool."""

    query: str = Field(
        ...,
        description=(
            "Natural language question about the data. Examples: "
            "'How many sales in January?', 'What is the average price by region?', "
            "'Show the top 10 stores by revenue', 'Filter rows where status is Valid'"
        ),
    )
    file_path: str = Field(
        ...,
        description=(
            "Path or URL to a CSV or XLSX file. "
            "Supports local file paths and public URLs (including OneDrive sharing links)."
        ),
    )


# ---------------------------------------------------------------------------
# File handling helpers
# ---------------------------------------------------------------------------


def _is_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _download_file(url: str) -> str:
    """Download a file from a URL to a temporary file and return its path."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; crewai-tools SpreadsheetAnalysis)",
    }
    response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    response.raise_for_status()

    # Determine extension from URL or content-type
    content_type = response.headers.get("Content-Type", "")
    if "spreadsheetml" in content_type or url.lower().endswith(".xlsx"):
        suffix = ".xlsx"
    else:
        suffix = ".csv"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV or XLSX file into a pandas DataFrame."""
    is_url = _is_url(file_path)
    local_path = file_path

    if is_url:
        local_path = _download_file(file_path)

    try:
        if local_path.lower().endswith(".xlsx") or local_path.lower().endswith(".xls"):
            df = pd.read_excel(local_path, engine="openpyxl")
        else:
            # Try common CSV encodings
            for encoding in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(local_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(local_path, encoding="utf-8", errors="replace")
        return df
    finally:
        if is_url:
            os.unlink(local_path)


# ---------------------------------------------------------------------------
# LLM code generation
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a pandas code generator. Given a DataFrame description and a natural language query, write Python code that answers the query.

Rules:
- The DataFrame is available as the variable `df`.
- `pd` (pandas) and `np` (numpy) are already imported.
- Write ONLY the code, wrapped in a single ```python``` code fence.
- The code MUST assign the final answer to a variable called `result`.
- `result` should be a DataFrame, Series, scalar, or string that directly answers the question.
- Do NOT import anything. Do NOT read/write files. Do NOT make network calls.
- Do NOT use exec(), eval(), __import__(), or os/sys modules.
- If the query asks for a count, use .shape[0] or .count() or len().
- If dates need filtering, try pd.to_datetime() on the relevant column first.
- Keep the code concise — ideally under 10 lines."""


def _build_user_prompt(query: str, df: pd.DataFrame) -> str:
    """Build the user prompt with DataFrame metadata and sample data."""
    buf = StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()

    sample = df.head(3).to_string(index=False)

    return f"""DataFrame info:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Columns and types:
{info_str}

First 3 rows:
{sample}

Query: {query}"""


def _extract_code(response: str) -> str:
    """Extract Python code from the LLM response."""
    # Try to find code in ```python ... ``` fences
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: try ``` ... ``` without language tag
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Last resort: use the whole response as code
    return response.strip()


# ---------------------------------------------------------------------------
# Safe execution
# ---------------------------------------------------------------------------

SAFE_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "range": range,
    "isinstance": isinstance,
    "type": type,
    "True": True,
    "False": False,
    "None": None,
    "print": print,
}

BLOCKED_PATTERNS = [
    r"\bimport\b",
    r"\b__import__\b",
    r"\bexec\b",
    r"\beval\b",
    r"\bopen\b",
    r"\bos\.",
    r"\bsys\.",
    r"\bsubprocess\b",
    r"\b__builtins__\b",
    r"\b__globals__\b",
    r"\bbreakpoint\b",
]


def _validate_code(code: str) -> str | None:
    """Check for disallowed patterns. Returns error message or None if safe."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"Generated code contains a blocked pattern: {pattern}"
    return None


def _safe_execute(code: str, df: pd.DataFrame) -> object:
    """Execute pandas code in a restricted namespace."""
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df,
    }
    exec(code, namespace)  # noqa: S102
    if "result" not in namespace:
        raise ValueError(
            "Generated code did not assign a value to 'result'. "
            "The LLM must set result = ..."
        )
    return namespace["result"]


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _format_result(result: object) -> str:
    """Convert a pandas/python result to a readable markdown string."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "The query returned an empty DataFrame (no matching rows)."
        if len(result) > 50:
            header = f"Showing first 50 of {len(result)} rows:\n\n"
            return header + result.head(50).to_markdown(index=False)
        return result.to_markdown(index=False)

    if isinstance(result, pd.Series):
        if result.empty:
            return "The query returned an empty Series (no matching data)."
        return result.to_markdown()

    return str(result)


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


class SpreadsheetAnalysis(BaseTool):
    name: str = "Spreadsheet Analysis Tool"
    description: str = (
        "Analyzes CSV and XLSX spreadsheet files using pandas. Converts natural "
        "language queries into pandas operations to perform aggregations (count, "
        "sum, average), filtering, grouping, sorting, and more. Use this tool "
        "instead of semantic search tools when you need to count rows, compute "
        "totals, filter by date/value, or perform any data analysis operation "
        "on tabular data."
    )
    args_schema: type[BaseModel] = SpreadsheetAnalysisInput
    llm_model: str = "openai/gpt-4o-mini"

    def _run(self, query: str, file_path: str) -> str:
        # 1. Load the data
        try:
            df = _load_dataframe(file_path)
        except Exception as e:
            return f"Error loading file: {e}"

        if df.empty:
            return "The file was loaded but contains no data."

        # 2. Build DataFrame summary for output
        columns_info = ", ".join(
            f"{col} ({df[col].dtype})" for col in df.columns
        )
        df_summary = (
            f"**DataFrame loaded:** {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"**Columns:** {columns_info}\n"
        )

        # 3. Generate pandas code from the query
        llm = LLM(model=self.llm_model)
        user_prompt = _build_user_prompt(query, df)

        try:
            response = llm.call(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as e:
            return f"{df_summary}\nError calling LLM for code generation: {e}"

        code = _extract_code(response)

        # 4. Validate the generated code
        validation_error = _validate_code(code)
        if validation_error:
            return (
                f"{df_summary}\n"
                f"Safety check failed: {validation_error}\n\n"
                f"**Generated code:**\n```python\n{code}\n```"
            )

        # 5. Execute the code
        try:
            result = _safe_execute(code, df)
        except Exception as e:
            return (
                f"{df_summary}\n"
                f"Error executing generated code: {e}\n\n"
                f"**Generated code:**\n```python\n{code}\n```"
            )

        # 6. Format and return the result with full context
        formatted = _format_result(result)
        return (
            f"**Query:** {query}\n\n"
            f"{df_summary}\n"
            f"**Generated pandas code:**\n```python\n{code}\n```\n\n"
            f"**Result:**\n{formatted}"
        )
