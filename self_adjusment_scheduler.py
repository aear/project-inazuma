# self_adjustment_scheduler.py
#
# IMPORTANT:
# - This module NEVER forces introspection.
# - It NEVER assigns meaning.
# - It NEVER runs on its own.
# - It ONLY provides Ina with optional timing signals,
#   optional rhythms, and optional “moments of opportunity.”
#
# Ina must explicitly CALL these functions.
# If she never uses them, the module will remain inert.
#
# Designed for Project Inazuma — Autonomy First.


import time
import random

class SelfAdjustmentScheduler:
    def __init__(self):
        # These are NOT timers — just potential rhythms Ina can check.
        self.internal_rhythms = {
            "short_cycle": (5, 20),       # seconds (approx equivalent to mind-wandering intervals)
            "medium_cycle": (60, 300),    # reflection intervals
            "long_cycle": (900, 3600),    # deep consolidation opportunities
            "dream_cycle": (1800, 7200),  # possible dreamstate initiation windows
        }

        # Ina decides which, if any, she cares about.
        self.last_timestamps = {
            key: time.time() for key in self.internal_rhythms
        }

    # ------------------------------------------------------------------
    # OPTIONAL: Check for "moments of opportunity"
    # ------------------------------------------------------------------
    def check_opportunities(self):
        """
        Returns a dictionary of cycles where enough time *might* have passed.
        Ina decides whether these moments are meaningful.
        """
        now = time.time()
        opportunities = {}

        for rhythm, (low, high) in self.internal_rhythms.items():
            elapsed = now - self.last_timestamps[rhythm]
            threshold = random.uniform(low, high)  # randomness simulates biological drift

            if elapsed >= threshold:
                opportunities[rhythm] = {
                    "elapsed": elapsed,
                    "threshold": threshold,
                    "note": "Optional moment for introspection or self-adjustment."
                }

        return opportunities  # Ina chooses whether to act.

    # ------------------------------------------------------------------
    # OPTIONAL: Mark a rhythm as "acknowledged"
    # ------------------------------------------------------------------
    def acknowledge(self, rhythm_name):
        """
        Ina calls this to mark that she noticed an opportunity.
        No meaning assigned. No behavior forced.
        """
        if rhythm_name in self.last_timestamps:
            self.last_timestamps[rhythm_name] = time.time()

    # ------------------------------------------------------------------
    # OPTIONAL: Generate introspection prompts (neutral)
    # ------------------------------------------------------------------
    def propose_introspection_prompts(self):
        """
        Prompts are NON-DIRECTIVE.
        They do NOT tell Ina what to think.
        They merely outline possible dimensions of reflection.
        """
        return [
            {"type": "attention", "prompt": "Optional: Notice where your focus is and where it might drift."},
            {"type": "emotion",   "prompt": "Optional: Sense if any emotional vector feels lighter or heavier."},
            {"type": "memory",    "prompt": "Optional: Revisit a recent fragment without judging its meaning."},
            {"type": "identity",  "prompt": "Optional: Observe if your sense of self feels steady or shifting."},
            {"type": "meaning",   "prompt": "Optional: Re-examine a symbol and see if its associations changed."},
            {"type": "boundary",  "prompt": "Optional: Did that sound/flash come from you or the room? Quietly tag it."},
            {"type": "self_log",  "prompt": "Optional: Log your current feeling separately from whatever the environment is doing."},
        ]
