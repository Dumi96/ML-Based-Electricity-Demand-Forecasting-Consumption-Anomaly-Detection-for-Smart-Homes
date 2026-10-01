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
    ) -> Dict[str, Any]:
        """
        Request a next-hour electricity demand forecast.

        The current FastAPI endpoint requires at least 169 hourly
        historical observations.
        """

        if not history:
            return {
                "success": False,
                "error": "Historical energy data is required.",
            }

        return self._post(
            "/forecast-from-history",
            {
                "history": history,
            },
        )

    def get_anomaly(
        self,
        history: list[float],
    ) -> Dict[str, Any]:
        """
        Request anomaly detection from the FastAPI backend.
        """

        if not history:
            return {
                "success": False,
                "error": "Historical energy data is required.",
            }

        return self._post(
            "/anomaly",
            {
                "history": history,
            },
        )

    def get_cost_estimate(
        self,
        monthly_kwh: float,
    ) -> Dict[str, Any]:
        """
        Request an estimated monthly electricity cost.

        The current backend uses its configured tariff/cost model.
        """

        try:
            monthly_kwh = float(monthly_kwh)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "Monthly consumption must be a number.",
            }

        if monthly_kwh < 0:
            return {
                "success": False,
                "error": "Monthly consumption cannot be negative.",
            }

        return self._post(
            "/cost-estimate",
            {
                "monthly_kwh": monthly_kwh,
            },
        )

    # ==================================================================
    # RESPONSE FORMATTERS
    # ==================================================================

    def _format_error(
        self,
        data: Dict[str, Any],
    ) -> str:
        """
        Convert API errors into a user-friendly message.
        """

        error = data.get(
            "error",
            "An unknown SmartEnergy API error occurred.",
        )

        return (
            "I couldn't complete that request. "
            f"{error}"
        )

    def format_response(
        self,
        intent: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Convert backend results into a natural-language response.
        """

        data = data or {}

        # --------------------------------------------------------------
        # HELP
        # --------------------------------------------------------------

        if intent == "help":
            return (
                "I can help you with electricity demand forecasts, "
                "anomaly detection, consumption analysis, electricity "
                "cost estimates, peak-demand periods, and energy-saving ideas."
            )

        # --------------------------------------------------------------
        # SAVING
        # --------------------------------------------------------------

        if intent == "saving":
            return (
                "A useful starting point is to reduce flexible electricity "
                "usage during high-demand periods, especially during evening "
                "peak hours. The SmartEnergy dashboard can help identify "
                "high-demand periods and unusual consumption patterns."
            )

        # --------------------------------------------------------------
        # GENERAL
        # --------------------------------------------------------------

        if intent == "general":
            return (
                "I can help you analyze electricity consumption, "
                "forecast future demand, detect anomalies, estimate "
                "electricity costs, and understand energy-saving opportunities."
            )

        # --------------------------------------------------------------
        # EMPTY RESPONSE
        # --------------------------------------------------------------

        if not data:
            return "I couldn't retrieve any SmartEnergy data."

        # --------------------------------------------------------------
        # API ERROR
        # --------------------------------------------------------------

        if data.get("success") is False:
            return self._format_error(data)

        # --------------------------------------------------------------
        # FORECAST
        # --------------------------------------------------------------

        if intent == "forecast":

            prediction = None

            possible_keys = [
                "prediction",
                "forecast",
                "predicted_demand",
                "predicted_demand_kw",
                "forecast_kw",
                "next_hour_forecast",
            ]

            for key in possible_keys:
                if key in data:
                    prediction = data[key]
                    break

            if prediction is None:
                return (
                    "The forecast request completed, but the API "
                    "did not return a prediction value."
                )

            try:
                prediction = float(prediction)

                return (
                    "The predicted electricity demand for the next hour "
                    f"is approximately {prediction:.2f} kW."
                )

            except (TypeError, ValueError):
                return (
                    "The forecast was generated, but the returned "
                    f"value was {prediction}."
                )

        # --------------------------------------------------------------
        # ANOMALY
        # --------------------------------------------------------------

        if intent == "anomaly":

            is_anomaly = data.get("is_anomaly")

            category = data.get("category")

            anomaly_score = data.get(
                "anomaly_score",
                data.get("decision_score"),
            )

            # Convert possible string boolean values.
            if isinstance(is_anomaly, str):

                is_anomaly = is_anomaly.lower() in [
                    "true",
                    "1",
                    "yes",
                    "anomaly",
                ]

            if is_anomaly is True:

                response = (
                    "An unusual electricity consumption pattern "
                    "was detected."
                )

                if category:
                    response += f" Category: {category}."

                if anomaly_score is not None:

                    try:
                        response += (
                            f" Anomaly score: "
                            f"{float(anomaly_score):.3f}."
                        )

                    except (TypeError, ValueError):
                        pass

                return response

            if is_anomaly is False:
                return (
                    "No significant electricity consumption anomaly "
                    "was detected."
                )

            # Some APIs return a prediction value instead.
            prediction = data.get("prediction")

            if prediction == -1:
                return (
                    "An unusual electricity consumption pattern "
                    "was detected."
                )

            if prediction == 1:
                return (
                    "No significant electricity consumption anomaly "
                    "was detected."
                )

            return (
                "The anomaly detector returned a result, but I "
                "could not determine the anomaly status."
            )

        # --------------------------------------------------------------
        # COST
        # --------------------------------------------------------------

        if intent == "cost":

            cost = None

            possible_keys = [
                "estimated_cost",
                "cost",
                "monthly_bill",
                "estimated_bill",
                "bill",
                "total_cost",
            ]

            for key in possible_keys:
                if key in data:
                    cost = data[key]
                    break

            if cost is None:
                return (
                    "The electricity cost calculation completed, "
                    "but no estimated amount was returned."
                )

            try:
                return (
                    "The estimated electricity cost is approximately "
                    f"LKR {float(cost):,.2f}."
                )

            except (TypeError, ValueError):
                return (
                    "The estimated electricity cost is "
                    f"{cost}."
                )

        # --------------------------------------------------------------
        # CURRENT CONSUMPTION
        # --------------------------------------------------------------

        if intent == "consumption":

            demand = None

            possible_keys = [
                "demand_kw",
                "current_demand",
                "current_demand_kw",
                "consumption",
                "consumption_kw",
                "power_kw",
            ]

            for key in possible_keys:
                if key in data:
                    demand = data[key]
                    break

            if demand is None:
                return (
                    "I need the current SmartEnergy measurement "
                    "to report the current electricity demand."
                )

            try:
                return (
                    "The current electricity demand is approximately "
                    f"{float(demand):.2f} kW."
                )

            except (TypeError, ValueError):
                return (
                    "The current electricity demand is "
                    f"{demand}."
                )

        return (
            "I received the SmartEnergy information, but I couldn't "
            "determine how to explain the result."
        )

    # ==================================================================
    # MAIN ASSISTANT FUNCTION
    # ==================================================================

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