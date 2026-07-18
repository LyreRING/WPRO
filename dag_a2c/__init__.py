"""Dual-DAG A2C research prototype."""

from dag_a2c.env import DualDAGServingEnv
from dag_a2c.generator import generate_dual_dag_instance

__all__ = ["DualDAGServingEnv", "generate_dual_dag_instance"]
