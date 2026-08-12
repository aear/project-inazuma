"""Modular cognition runtime behind the stable Model Manager façade."""
from .capability_registry import CapabilityRegistry
from .cognitive_context import CognitiveContext, ContextReference
from .contracts import CapabilitySpec, Contribution, CostEstimate
from .live_patch import HandlerGeneration, LivePatchManager
from .resource_budget import BudgetDecision, BudgetSnapshot, ResourceBudget
from .result_bus import ResultBus
from .runtime import CognitionRuntime
from .scheduler import ExistingSchedulerAdapter, capability_specs_from_task_profiles

__all__ = [
    "BudgetDecision", "BudgetSnapshot", "CapabilityRegistry", "CapabilitySpec",
    "CognitionRuntime", "CognitiveContext", "ContextReference", "Contribution",
    "CostEstimate", "ExistingSchedulerAdapter", "HandlerGeneration", "LivePatchManager",
    "ResourceBudget", "ResultBus", "capability_specs_from_task_profiles",
]
