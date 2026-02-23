"""Curriculum training pipeline for MohaNetLight.

Submodules
----------
curriculum : CurriculumTrainer, PhaseConfig, default_curriculum
report     : generate_full_report, generate_phase_report, generate_combined_report
"""

from train.curriculum import CurriculumTrainer, PhaseConfig, default_curriculum
from train.report import generate_combined_report, generate_full_report, generate_phase_report

__all__ = [
    "CurriculumTrainer",
    "PhaseConfig",
    "default_curriculum",
    "generate_full_report",
    "generate_phase_report",
    "generate_combined_report",
]
