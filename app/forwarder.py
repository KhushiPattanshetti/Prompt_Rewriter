"""
forwarder.py
Forwards the validated clinical note to the next pipeline block:
  Prompt Rewriter Block (Phi-3 Mini SLM)

In development/test mode (PROMPT_REWRITER_URL not set), the forward is
simulated and logged instead of making a real HTTP call.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

PROMPT_REWRITER_URL = os.environ.get("PROMPT_REWRITER_URL", "")


def forward_to_prompt_rewriter(
    filename: str,
    file_content: str,
    instruction_id: str,
) -> dict:
    """
    Forward clinical note content to the Prompt Rewriter Block.

    Returns a dict with:
      - forwarded: bool
      - destination: str
      - instruction_id: str
      - simulated: bool (True when no real endpoint configured)
    """
    payload = {
        "instruction_id": instruction_id,
        "filename": filename,
        "clinical_note": file_content,
    }

    if not PROMPT_REWRITER_URL:
        # ── Simulation mode: log and return mock response ─────────────────
        logger.info(
            "[FORWARDER] Simulated forward to Prompt Rewriter. "
            "Set PROMPT_REWRITER_URL env var to enable real forwarding. "
            "Payload keys: %s",
            list(payload.keys()),
        )
        return {
            "forwarded": True,
            "destination": "simulated (PROMPT_REWRITER_URL not configured)",
            "instruction_id": instruction_id,
            "simulated": True,
            "payload_preview": {
                "filename": filename,
                "note_length": len(file_content),
                "instruction_id": instruction_id,
            },
        }

    # ── Real HTTP POST to Prompt Rewriter Block ───────────────────────────
    try:
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            PROMPT_REWRITER_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            response_body = response.read().decode("utf-8")
            logger.info("[FORWARDER] Successfully forwarded to %s", PROMPT_REWRITER_URL)
            return {
                "forwarded": True,
                "destination": PROMPT_REWRITER_URL,
                "instruction_id": instruction_id,
                "simulated": False,
                "response": response_body,
            }
    except Exception as exc:
        logger.error("[FORWARDER] Failed to forward: %s", str(exc))
        return {
            "forwarded": False,
            "destination": PROMPT_REWRITER_URL,
            "instruction_id": instruction_id,
            "simulated": False,
            "error": str(exc),
        }
