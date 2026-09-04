"""BP+OSD list decoder with action-level posterior aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from bb64_hybrid_nonclifford.model import HybridModel

from .action_model import RepairActionBuilder, ScheduledRepairAction
from .belief_propagation import BPResult, CategoricalBPDecoder
from .factor_graph import D3FactorGraph
from .osd import FaultHypothesis, OSDResult, osd_candidate_list


@dataclass(frozen=True)
class ActionPosteriorEntry:
    action: ScheduledRepairAction
    list_probability: float
    log_weight: float
    explanation_count: int


@dataclass(frozen=True)
class D3DecodeResult:
    reset: bool
    reset_reason: str | None
    action: ScheduledRepairAction | None
    leading_list_probability: float
    runner_up_list_probability: float
    log_likelihood_gap: float
    candidate_probability_mass: float
    candidates: int
    distinct_actions: int
    bp: BPResult
    osd: OSDResult
    action_posteriors: tuple[ActionPosteriorEntry, ...]


def _logadd(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


class D3ListDecoder:
    """Decode an arbitrary-weight detector history with BP and an OSD list."""

    def __init__(
        self,
        graph: D3FactorGraph,
        model: HybridModel,
        *,
        theta: float,
        bp_iterations: int = 40,
        bp_damping: float = 0.45,
        osd_order: int = 2,
        osd_window: int = 64,
        maximum_candidates: int = 512,
        minimum_action_probability: float = 0.99,
        require_bp_convergence: bool = False,
    ):
        if not 0.0 <= minimum_action_probability <= 1.0:
            raise ValueError("minimum action probability must lie in [0,1]")
        self.graph = graph
        self.model = model
        self.action_builder = RepairActionBuilder(model, theta=theta)
        self.bp_decoder = CategoricalBPDecoder(
            graph,
            maximum_iterations=bp_iterations,
            damping=bp_damping,
        )
        self.osd_order = int(osd_order)
        self.osd_window = int(osd_window)
        self.maximum_candidates = int(maximum_candidates)
        self.minimum_action_probability = float(minimum_action_probability)
        self.require_bp_convergence = bool(require_bp_convergence)

    def action_for_hypothesis(
        self, candidate: FaultHypothesis
    ) -> ScheduledRepairAction:
        action, _branch_probability = self.action_builder.from_signature(
            candidate.branch_class_id,
            final_x_mask=candidate.physical_signature.final_x_mask,
            final_z_mask=candidate.physical_signature.final_z_mask,
            rotation_sign_mask=candidate.physical_signature.rotation_sign_mask,
        )
        return action

    def decode(self, observed_detector_mask: int) -> D3DecodeResult:
        bp = self.bp_decoder.decode(observed_detector_mask)
        osd = osd_candidate_list(
            self.graph,
            self.model,
            observed_detector_mask,
            bp,
            order=self.osd_order,
            free_window=self.osd_window,
            maximum_candidates=self.maximum_candidates,
        )
        action_logs: dict[ScheduledRepairAction, tuple[float, int]] = {}
        for candidate in osd.candidates:
            action = self.action_for_hypothesis(candidate)
            previous, count = action_logs.get(action, (-math.inf, 0))
            action_logs[action] = (_logadd(previous, candidate.log_probability), count + 1)
        if not action_logs:
            return D3DecodeResult(
                True,
                "no_exact_syndrome_candidate",
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                bp,
                osd,
                (),
            )
        normalization = -math.inf
        for log_weight, _count in action_logs.values():
            normalization = _logadd(normalization, log_weight)
        entries = sorted(
            (
                ActionPosteriorEntry(
                    action,
                    math.exp(log_weight - normalization),
                    log_weight,
                    count,
                )
                for action, (log_weight, count) in action_logs.items()
            ),
            key=lambda entry: (-entry.list_probability, entry.log_weight),
        )
        leading = entries[0]
        runner = entries[1].list_probability if len(entries) > 1 else 0.0
        gap = (
            leading.log_weight - entries[1].log_weight
            if len(entries) > 1
            else math.inf
        )
        reason = None
        if self.require_bp_convergence and not bp.converged:
            reason = "bp_not_converged"
        elif leading.list_probability < self.minimum_action_probability:
            reason = "list_action_probability"
        maximum = max(candidate.log_probability for candidate in osd.candidates)
        mass = math.exp(maximum) * math.fsum(
            math.exp(candidate.log_probability - maximum) for candidate in osd.candidates
        )
        return D3DecodeResult(
            reset=reason is not None,
            reset_reason=reason,
            action=None if reason is not None else leading.action,
            leading_list_probability=leading.list_probability,
            runner_up_list_probability=runner,
            log_likelihood_gap=gap,
            candidate_probability_mass=mass,
            candidates=len(osd.candidates),
            distinct_actions=len(entries),
            bp=bp,
            osd=osd,
            action_posteriors=tuple(entries),
        )
