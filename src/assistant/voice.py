"""
SmartEnergy AI - Voice Assistant

Handles:
1. Microphone input
2. Speech-to-text
3. Text-to-speech

Required packages:
    pip install SpeechRecognition pyttsx3 PyAudio
"""

from __future__ import annotations

import re
import threading
from typing import Optional

import pyttsx3
import speech_recognition as sr


class VoiceAssistant:
    """
    Voice input and output handler for SmartEnergy AI.
    """

    def __init__(
        self,
        language: str = "en-US",
        microphone_index: Optional[int] = None,
    ):
        self.language = language

        # If None, PyAudio/SpeechRecognition will use
        # the Windows default microphone.
        self.microphone_index = microphone_index

        # Speech recognition
        self.recognizer = sr.Recognizer()

        # Prevent multiple TTS engines running together
        self._speech_lock = threading.Lock()

        # We intentionally do NOT keep a pyttsx3 engine alive.
        # Streamlit reruns can cause "run loop already started".
        self.engine = None

        # Recognition settings
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    # ============================================================
    # MICROPHONE LIST
    # ============================================================

    @staticmethod
    def list_microphones() -> list[str]:
        """
        Return all microphones detected by SpeechRecognition.
        """

        try:
            microphones = sr.Microphone.list_microphone_names()
            return [
                f"{index}: {name}"
                for index, name in enumerate(microphones)
            ]

        except Exception as error:
            print(f"Could not list microphones: {error}")
            return []

    # ============================================================
    # MICROPHONE TEST
    # ============================================================

    def test_microphone(self) -> bool:
        """
        Test whether the selected microphone can be opened.
        """

        try:
            with sr.Microphone(
                device_index=self.microphone_index
            ) as source:

                print("Microphone opened successfully.")

                print(
                    f"Microphone index: "
                    f"{self.microphone_index}"
                )

                print(
                    "Please stay quiet for 1 second "
                    "while the microphone is calibrated..."
                )

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1,
                )

                print(
                    "Calibration completed."
                )

                print(
                    "Energy threshold:",
                    self.recognizer.energy_threshold,
                )

                return True

        except Exception as error:
            print(
                "Microphone test failed:"
            )

            print(error)

            return False

    # ============================================================
    # LISTEN
    # ============================================================

    def listen(
        self,
        timeout: int = 15,
        phrase_time_limit: int = 15,
    ) -> str:
        """
        Listen to microphone and convert speech to text.

        Returns:
            Recognized text, or empty string if unsuccessful.
        """

        print()
        print("=" * 60)
        print("SMARTENERGY AI VOICE INPUT")
        print("=" * 60)

        # --------------------------------------------------------
        # Open microphone
        # --------------------------------------------------------

        try:

            with sr.Microphone(
                device_index=self.microphone_index
            ) as source:

                print()
                print("Microphone connected.")

                # ------------------------------------------------
                # Calibrate microphone
                # ------------------------------------------------

                print(
                    "Adjusting microphone for background noise..."
                )

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=1,
                )

                print(
                    "Energy threshold:",
                    round(
                        self.recognizer.energy_threshold,
                        2,
                    ),
                )

                # ------------------------------------------------
                # Listen
                # ------------------------------------------------

                print()
                print(
                    "🎤 LISTENING NOW..."
                )

                print(
                    "Speak your question."
                )

                print(
                    "Example: "
                    "What is my next hour forecast?"
                )

                print()

                try:

                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit,
                    )

                except sr.WaitTimeoutError:

                    print()
                    print(
                        "No speech detected "
                        f"within {timeout} seconds."
                    )

                    return ""

                except Exception as listen_error:

                    print()
                    print(
                        "Listening error:"
                    )

                    print(listen_error)

                    return ""

        # --------------------------------------------------------
        # Microphone opening error
        # --------------------------------------------------------

        except Exception as microphone_error:

            print()
            print(
                "MICROPHONE ERROR"
            )

            print(
                microphone_error
            )

            return ""

        # --------------------------------------------------------
        # Speech recognition
        # --------------------------------------------------------

        print()
        print(
            "Audio captured."
        )

        print(
            "Converting speech to text..."
        )

        try:

            text = self.recognizer.recognize_google(
                audio,
                language=self.language,
            )

            text = text.strip()

            if text:

                print()
                print(
                    "Recognized speech:"
                )

                print(
                    text
                )

                print()

                return text

            print(
                "Speech recognition returned empty text."
            )

            return ""

        except sr.UnknownValueError:

            print()
            print(
                "Could not understand the speech."
            )

            return ""

        except sr.RequestError as error:

            print()
            print(
                "Google speech recognition service "
                "could not be reached."
            )

            print(error)

            return ""

        except Exception as error:

            print()
            print(
                "Speech recognition error:"
            )

            print(error)

            return ""

    # ============================================================
    # CLEAN TEXT FOR SPEECH
    # ============================================================

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """
        Remove Markdown/HTML formatting before TTS.
        """

        if not text:
            return ""

        clean = text

        # Markdown bold
        clean = clean.replace("**", "")

        # Markdown italic
        clean = clean.replace("__", "")

        # Inline code
        clean = clean.replace("`", "")

        # HTML tags
        clean = re.sub(
            r"<[^>]+>",
            "",
            clean,
        )

        # Markdown headings
        clean = re.sub(
            r"^#+\s*",
            "",
            clean,
            flags=re.MULTILINE,
        )

        # Bullet points
        clean = re.sub(
            r"^\s*[-*]\s*",
            "",
            clean,
            flags=re.MULTILINE,
        )

        # Extra spaces
        clean = re.sub(
            r"\s+",
            " ",
            clean,
        )

        return clean.strip()

    # ============================================================
    # SPEAK
    # ============================================================

    def speak(
        self,
        text: str,
        asynchronous: bool = False,
    ) -> None:
        """
        Convert text to speech.

        A new pyttsx3 engine is created for each request.
        This prevents Streamlit/pyttsx3 event-loop conflicts.
        """

        if not text:
            return

        clean_text = self.clean_text_for_speech(
            text
        )

        if not clean_text:
            return

        def _speak():

            with self._speech_lock:

                engine = None

                try:

                    print()
                    print(
                        "🔊 SmartEnergy AI:"
                    )

                    print(
                        clean_text
                    )

                    # IMPORTANT:
                    # Create a fresh engine every time.
                    engine = pyttsx3.init()

                    # Speech speed
                    engine.setProperty(
                        "rate",
                        165,
                    )

                    # Volume
                    engine.setProperty(
                        "volume",
                        1.0,
                    )

                    # Speak
                    engine.say(
                        clean_text
                    )

                    engine.runAndWait()

                except Exception as error:

                    print()
                    print(
                        "Text-to-speech error:"
                    )

                    print(error)

                finally:

                    if engine is not None:

                        try:
                            engine.stop()

                        except Exception:
                            pass

        # --------------------------------------------------------
        # Synchronous mode
        # --------------------------------------------------------

        if not asynchronous:

            _speak()

            return

        # --------------------------------------------------------
        # Asynchronous mode
        # --------------------------------------------------------

        thread = threading.Thread(
            target=_speak,
            daemon=True,
        )

        thread.start()

    # ============================================================
    # LISTEN + RESPOND
    # ============================================================

    def listen_and_respond(
        self,
        assistant,
        data: Optional[dict] = None,
    ) -> tuple[str, str]:

        # Listen
        question = self.listen()

        # --------------------------------------------------------
        # No speech
        # --------------------------------------------------------

        if not question:

            response = (
                "I couldn't understand your voice input. "
                "Please try again."
            )

            self.speak(
                response
            )

            return "", response

        # --------------------------------------------------------
        # AI response
        # --------------------------------------------------------

        try:

            response = assistant.ask(
                question,
                data=data,
            )

        except Exception as error:

            print(
                "Assistant error:"
            )

            print(error)

            response = (
                "I received your question, "
                "but I could not generate an answer "
                "right now."
            )

        # --------------------------------------------------------
        # Speak response
        # --------------------------------------------------------

        self.speak(
            response
        )

        return question, response


# ================================================================
# STANDALONE TEST
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SMARTENERGY AI - VOICE ASSISTANT TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Show microphones
    # ------------------------------------------------------------

    print()
    print("Available microphones:")
    print("-" * 70)

    microphones = VoiceAssistant.list_microphones()

    if microphones:

        for microphone in microphones:
            print(microphone)

    else:

        print(
            "No microphones detected."
        )

    # ------------------------------------------------------------
    # Create assistant
    # ------------------------------------------------------------

    voice = VoiceAssistant()

    # ------------------------------------------------------------
    # Test TTS
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST 1 - SPEAKER")
    print("=" * 70)

    voice.speak(
        "Hello. Smart Energy AI voice output is working."
    )

    # ------------------------------------------------------------
    # Test microphone
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST 2 - MICROPHONE")
    print("=" * 70)

    print()
    print(
        "When you see LISTENING NOW, say:"
    )

    print(
        "What is my next hour forecast?"
    )

    result = voice.listen(
        timeout=15,
        phrase_time_limit=15,
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if result:

        print(
            "SUCCESS:"
        )

        print(
            result
        )

    else:

        print(
            "No speech was recognized."
        )