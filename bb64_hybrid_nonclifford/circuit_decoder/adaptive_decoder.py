"""Fail-closed Stage D3 controller combining BP, OSD, and posterior MCMC."""

from __future__ import annotations

from dataclasses import dataclass

from .action_model import ScheduledRepairAction
from .list_decoder import D3DecodeResult, D3ListDecoder
from .mcmc import MCMCPosterior, sample_action_posterior


@dataclass(frozen=True)
class D3AdaptiveResult:
    reset: bool
    reset_reason: str | None
    action: ScheduledRepairAction | None
    list_result: D3DecodeResult
    posterior: MCMCPosterior | None


class D3AdaptiveDecoder:
    """Return an action only when independently sampled posterior checks pass."""

    def __init__(
        self,
        list_decoder: D3ListDecoder,
        *,
        minimum_posterior_lower_bound: float = 0.99,
        minimum_effective_samples: float = 100.0,
        maximum_chain_spread: float = 0.10,
        minimum_action_transitions: int = 1,
        chains: int = 4,
        burn_in: int = 1000,
        retained_per_chain: int = 500,
        thin: int = 5,
        seed: int = 20260904,
    ):
        if not 0.0 <= minimum_posterior_lower_bound <= 1.0:
            raise ValueError("minimum posterior lower bound must lie in [0,1]")
        if minimum_effective_samples < 0 or not 0.0 <= maximum_chain_spread <= 1.0:
            raise ValueError("invalid posterior diagnostic threshold")
        if minimum_action_transitions < 0:
            raise ValueError("minimum action transitions must be nonnegative")
        self.list_decoder = list_decoder
        self.minimum_posterior_lower_bound = float(minimum_posterior_lower_bound)
        self.minimum_effective_samples = float(minimum_effective_samples)
        self.maximum_chain_spread = float(maximum_chain_spread)
        self.minimum_action_transitions = int(minimum_action_transitions)
        self.chains = int(chains)
        self.burn_in = int(burn_in)
        self.retained_per_chain = int(retained_per_chain)
        self.thin = int(thin)
        self.seed = int(seed)
        self.decode_count = 0

    def decode(self, observed_detector_mask: int) -> D3AdaptiveResult:
        listed = self.list_decoder.decode(observed_detector_mask)
        if not listed.osd.candidates:
            return D3AdaptiveResult(True, "no_exact_syndrome_candidate", None, listed, None)
        starts = []
        actions = set()
        for candidate in listed.osd.candidates:
            action = self.list_decoder.action_for_hypothesis(candidate)
            if action not in actions:
                actions.add(action)
                starts.append(candidate)
            if len(starts) >= self.chains:
                break
        posterior = sample_action_posterior(
            self.list_decoder.graph,
            self.list_decoder.model,
            self.list_decoder.action_builder,
            observed_detector_mask,
            listed.bp,
            starts,
            chains=self.chains,
            burn_in=self.burn_in,
            retained_per_chain=self.retained_per_chain,
            thin=self.thin,
            seed=self.seed + self.decode_count,
        )
        self.decode_count += 1
        reason = None
        if posterior.total_action_transitions < self.minimum_action_transitions:
            reason = "insufficient_action_mixing"
        elif posterior.effective_sample_size < self.minimum_effective_samples:
            reason = "insufficient_effective_samples"
        elif posterior.maximum_chain_fraction_spread > self.maximum_chain_spread:
            reason = "chain_disagreement"
        elif posterior.leading_probability_lower_bound < self.minimum_posterior_lower_bound:
            reason = "posterior_lower_bound"
        return D3AdaptiveResult(
            reset=reason is not None,
            reset_reason=reason,
            action=None if reason is not None else posterior.leading_action,
            list_result=listed,
            posterior=posterior,
        )
