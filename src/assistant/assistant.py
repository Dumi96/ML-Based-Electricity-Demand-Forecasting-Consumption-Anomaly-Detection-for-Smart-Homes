"""
SmartEnergy AI Assistant

Handles:
- User question intent detection
- FastAPI communication
- Forecast requests
- Anomaly detection requests
- Electricity cost estimation
- Current consumption responses
- Energy-saving guidance

This module is intentionally independent from Streamlit so it can
later be used by the web dashboard and voice assistant.
"""

from __future__ import annotations
from src.inference import forecast_from_history, detect_anomaly, estimate_cost

from typing import Any, Dict, Optional

import requests

from .prompts import build_response_prompt


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEFAULT_API_URL = "http://127.0.0.1:8000"


class SmartEnergyAssistant:
    """
    Main SmartEnergy assistant class.

    The assistant currently uses lightweight intent detection.
    Later, this can be replaced or enhanced with an LLM.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
    ):
        self.api_url = api_url.rstrip("/")

    # ==================================================================
    # HTTP HELPERS
    # ==================================================================

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a GET request to the FastAPI backend.
        """

        try:
            response = requests.get(
                f"{self.api_url}{endpoint}",
                params=params,
                timeout=10,
            )

            response.raise_for_status()

            result = response.json()

            if isinstance(result, dict):
                return result

            return {
                "success": True,
                "data": result,
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Unable to connect to the SmartEnergy API.",
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "The SmartEnergy API request timed out.",
            }

        except requests.exceptions.HTTPError as exc:
            return {
                "success": False,
                "error": f"API returned an HTTP error: {exc}",
            }

        except requests.exceptions.RequestException as exc:
            return {
                "success": False,
                "error": str(exc),
            }

        except ValueError:
            return {
                "success": False,
                "error": "The API returned an invalid JSON response.",
            }

    def _post(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send a POST request to the FastAPI backend.
        """

        try:
            response = requests.post(
                f"{self.api_url}{endpoint}",
                json=payload,
                timeout=15,
            )

            response.raise_for_status()

            result = response.json()

            if isinstance(result, dict):
                return result

            return {
                "success": True,
                "data": result,
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Unable to connect to the SmartEnergy API.",
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "The SmartEnergy API request timed out.",
            }

        except requests.exceptions.HTTPError as exc:
            return {
                "success": False,
                "error": f"API returned an HTTP error: {exc}",
            }

        except requests.exceptions.RequestException as exc:
            return {
                "success": False,
                "error": str(exc),
            }

        except ValueError:
            return {
                "success": False,
                "error": "The API returned an invalid JSON response.",
            }

    # ==================================================================
    # INTENT DETECTION
    # ==================================================================

    def detect_intent(self, question: str) -> str:
        """
        Detect the main intent of a user question.

        Current implementation uses keyword-based classification.
        This can later be upgraded to an LLM-based intent router.
        """

        if not question:
            return "general"

        text = question.lower().strip()

        # --------------------------------------------------------------
        # HELP
        # --------------------------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "what can you do",
                "what can you help",
                "how can you help",
                "help me",
                "help",
                "available features",
                "your features",
                "your capabilities",
            ]
        ):
            return "help"

        # --------------------------------------------------------------
        # FORECAST
        # --------------------------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "forecast",
                "predict",
                "prediction",
                "future demand",
                "future consumption",
                "next hour",
                "next-hour",
                "tomorrow demand",
                "expected demand",
            ]
        ):
            return "forecast"

        # --------------------------------------------------------------
        # ANOMALY
        # --------------------------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "anomaly",
                "anomalies",
                "abnormal",
                "abnormality",
                "unusual",
                "unusual consumption",
                "sudden change",
                "sudden increase",
                "sudden decrease",
                "something wrong",
            ]
        ):
            return "anomaly"

        # --------------------------------------------------------------
        # COST
        # --------------------------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "cost",
                "electricity cost",
                "electricity bill",
                "energy bill",
                "bill",
                "tariff",
                "electricity price",
                "how much will",
                "how much would",
            ]
        ):
            return "cost"

        # --------------------------------------------------------------
        # SAVING / EFFICIENCY
        # --------------------------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "save electricity",
                "save energy",
                "saving energy",
                "energy saving",
                "reduce consumption",
                "reduce electricity",
                "reduce energy",
                "lower consumption",
                "lower electricity",
                "energy efficiency",
                "efficient",
                "efficiency",
            ]
        ):
            return "saving"

        # --------------------------------------------------------------
        # CURRENT CONSUMPTION
        # --------------------------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "current consumption",
                "current demand",
                "current energy",
                "current usage",
                "right now",
                "currently using",
                "how much electricity",
                "how much energy",
                "what am i using",
                "what is my consumption",
                "what is the current",
            ]
        ):
            return "consumption"

        # --------------------------------------------------------------
        # GENERAL CONSUMPTION QUESTIONS
        # --------------------------------------------------------------

        if any(
            word in text
            for word in [
                "consumption",
                "usage",
                "demand",
                "energy",
                "electricity",
            ]
        ):
            return "consumption"

        return "general"

    # ==================================================================
    # API OPERATIONS
    # ==================================================================

    def get_health(self) -> Dict[str, Any]:
        """
        Check whether the FastAPI backend is available.
        """

        return self._get("/health")

    def get_forecast(
        self,
        history: list[float],
        history_end_timestamp: str | None = None,
    ) -> Dict[str, Any]:
        """Forecast the next hour using the API or local model."""
        if len(history) < 169:
            return {
                "success": False,
                "error": "At least 169 hourly observations are required.",
            }

        from datetime import datetime, timezone

        timestamp = history_end_timestamp or datetime.now(
            timezone.utc
        ).replace(tzinfo=None).isoformat()

        payload = {
            "history_end_timestamp": timestamp,
            "hourly_consumption_kw": list(history),
        }

        result = self._post("/forecast-from-history", payload)
        if result and not result.get("error") and result.get("success") is not False:
            return result

        try:
            return forecast_from_history(timestamp, list(history))
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            return {"success": False, "error": str(exc)}

    def get_anomaly(
        self,
        history: list[float],
        history_end_timestamp: str | None = None,
    ) -> Dict[str, Any]:
        """Detect consumption anomalies using the API or local model."""
        if len(history) < 169:
            return {
                "success": False,
                "error": "At least 169 hourly observations are required.",
            }

        from datetime import datetime, timezone

        timestamp = history_end_timestamp or datetime.now(
            timezone.utc
        ).replace(tzinfo=None).isoformat()

        payload = {
            "history_end_timestamp": timestamp,
            "hourly_consumption_kw": list(history),
        }

        result = self._post("/anomaly", payload)
        if result and not result.get("error") and result.get("success") is not False:
            return result

        try:
            return detect_anomaly(timestamp, list(history))
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            return {"success": False, "error": str(exc)}

    def get_cost_estimate(
        self,
        monthly_kwh: float,
    ) -> Dict[str, Any]:
        """Estimate monthly electricity cost using API or local reference data."""
        try:
            monthly_kwh = float(monthly_kwh)
        except (TypeError, ValueError):
            return {"success": False, "error": "Monthly consumption must be a number."}

        if monthly_kwh < 0:
            return {"success": False, "error": "Monthly consumption cannot be negative."}

        payload = {"monthly_energy_kwh": monthly_kwh}
        result = self._post("/cost-estimate", payload)

        if result and not result.get("error") and result.get("success") is not False:
            return result

        try:
            return estimate_cost(monthly_kwh)
        except (ValueError, RuntimeError) as exc:
            return {"success": False, "error": str(exc)}

    def ask(
        self,
        question: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process a user question.

        Parameters
        ----------
        question:
            User's natural-language question.

        data:
            Optional dashboard data.

            Examples:

                {
                    "history": [...]
                }

            or:

                {
                    "demand_kw": 1.82
                }

            or:

                {
                    "monthly_kwh": 245.5
                }

        Returns
        -------
        str
            Natural-language assistant response.
        """

        # --------------------------------------------------------------
        # Validate question
        # --------------------------------------------------------------

        if not question or not question.strip():
            return (
                "Please ask me an electricity-related question."
            )

        question = question.strip()

        # --------------------------------------------------------------
        # Detect intent
        # --------------------------------------------------------------

        intent = self.detect_intent(question)

        # --------------------------------------------------------------
        # Build context for future LLM integration
        # --------------------------------------------------------------

        context = ""

        if data:
            context = str(data)

        # This currently prepares the prompt architecture for a future
        # LLM integration. We don't call an external LLM here yet.
        _ = build_response_prompt(
            question=question,
            context=context,
        )

        # --------------------------------------------------------------
        # Help
        # --------------------------------------------------------------

        if intent == "help":
            return self.format_response(
                intent="help",
                data={},
            )

        # --------------------------------------------------------------
        # Energy saving
        # --------------------------------------------------------------

        if intent == "saving":
            return self.format_response(
                intent="saving",
                data={},
            )

        # --------------------------------------------------------------
        # Forecast
        # --------------------------------------------------------------

        if intent == "forecast":

            if not data or "history" not in data:
                return (
                    "I can generate the next-hour forecast, but I need "
                    "the historical hourly energy data from the dashboard."
                )

            history = data["history"]

            if not isinstance(history, (list, tuple)):
                return (
                    "The historical energy data format is invalid."
                )

            if len(history) < 169:
                return (
                    "I need at least 169 hourly energy observations "
                    "to generate the forecast."
                )

            result = self.get_forecast(
                history=list(history),
                history_end_timestamp=data.get("history_end_timestamp"),
            )

            return self.format_response(
                intent="forecast",
                data=result,
            )

        # --------------------------------------------------------------
        # Anomaly
        # --------------------------------------------------------------

        if intent == "anomaly":

            if not data or "history" not in data:
                return (
                    "I can check for consumption anomalies, but I need "
                    "the historical hourly energy data from the dashboard."
                )

            history = data["history"]

            if not isinstance(history, (list, tuple)):
                return (
                    "The historical energy data format is invalid."
                )

            if len(history) < 169:
                return (
                    "I need at least 169 hourly energy observations "
                    "to perform anomaly detection."
                )

            result = self.get_anomaly(
                history=list(history),
                history_end_timestamp=data.get("history_end_timestamp"),
            )

            return self.format_response(
                intent="anomaly",
                data=result,
            )

        # --------------------------------------------------------------
        # Cost
        # --------------------------------------------------------------

        if intent == "cost":

            if not data or "monthly_kwh" not in data:
                return (
                    "I can estimate your electricity cost, but I need "
                    "your monthly energy consumption in kWh."
                )

            try:
                monthly_kwh = float(
                    data["monthly_kwh"]
                )

            except (TypeError, ValueError):
                return (
                    "Please provide the monthly electricity "
                    "consumption as a number in kWh."
                )

            result = self.get_cost_estimate(
                monthly_kwh=monthly_kwh,
            )

            return self.format_response(
                intent="cost",
                data=result,
            )

        # --------------------------------------------------------------
        # Current consumption
        # --------------------------------------------------------------

        if intent == "consumption":

            if data and "demand_kw" in data:

                return self.format_response(
                    intent="consumption",
                    data={
                        "demand_kw": data["demand_kw"],
                    },
                )

            return (
                "I need the current SmartEnergy measurement to tell "
                "you the current electricity demand."
            )

        # --------------------------------------------------------------
        # General question
        # --------------------------------------------------------------

        return self.format_response(
            intent="general",
            data={},
        )


# ======================================================================
# SIMPLE STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    assistant = SmartEnergyAssistant()

    test_questions = [
        "What can you do?",
        "How can I save electricity?",
        "What is the next hour forecast?",
        "Are there any anomalies?",
        "What is my current consumption?",
        "How much will my electricity bill cost?",
    ]

    print("=" * 70)
    print("SmartEnergy AI Assistant Test")
    print("=" * 70)

    for question in test_questions:

        print()
        print(f"User: {question}")

        response = assistant.ask(question)

        print(f"Assistant: {response}")

    print()
    print("=" * 70)
