# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from twinkle.data_format import Trajectory
from twinkle_agentic.tools.tool_manager import ToolManager
from .data_plane import TransferQueueDataPlane
from .registry import AdapterRegistry
from .scheduling import PreferCurrentTrainPolicy, WorkConservingRolloutPolicy
from .staleness import StalenessManager
from .types import (PartialRolloutConfig, PartitionMetadata, PartitionStatus, RolloutContextState, RolloutGroupRequest,
                    RolloutGroupResult, SampleRecord, TrainingContext)


class ToolManagerFactory:
    """Create context-scoped ToolManager instances.

    Profiles are callables so deployments can attach native or remote tools
    without importing untrusted user code in the server process.
    """

    def __init__(self, profiles: dict[str, Callable[[TrainingContext, SampleRecord], ToolManager]] | None = None):
        self._profiles = dict(profiles or {})

    def register(self, profile: str, factory: Callable[[TrainingContext, SampleRecord], ToolManager]) -> None:
        self._profiles[profile] = factory

    def create(self, sample: SampleRecord, context: TrainingContext) -> ToolManager:
        factory = self._profiles.get(context.tool_profile)
        if factory is None:
            return ToolManager()
        return factory(context, sample)


class AsyncRollouter:
    """Prompt-group producer with per-context gating and pluggable scheduling."""

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        adapter_registry: AdapterRegistry,
        staleness_manager: StalenessManager,
        rollout,
        tool_manager_factory: ToolManagerFactory | None = None,
        rollout_policy: Any | None = None,
        partial_rollout_config: PartialRolloutConfig | None = None,
        max_concurrent_groups: int = 16,
        target_groups_per_partition: int = 1,
    ):
        self.data_plane = data_plane
        self.adapter_registry = adapter_registry
        self.staleness_manager = staleness_manager
        self.rollout = rollout
        self.tool_manager_factory = tool_manager_factory or ToolManagerFactory()
        self.rollout_policy = rollout_policy or WorkConservingRolloutPolicy()
        self.partial_rollout_config = partial_rollout_config or PartialRolloutConfig()
        self.max_concurrent_groups = max_concurrent_groups
        self.target_groups_per_partition = target_groups_per_partition
        self.pending_by_context: dict[str, Deque[RolloutGroupRequest]] = defaultdict(deque)
        self.active_tasks: set[asyncio.Task] = set()
        self.transfer_buffer_by_context: dict[str, list[Trajectory]] = defaultdict(list)
        self._last_submit_time: dict[str, float] = defaultdict(float)
        self._submitted_groups: dict[str, int] = defaultdict(int)

    def add_pending(self, context: TrainingContext, samples: Iterable[SampleRecord]) -> None:
        self.adapter_registry.register(context)
        self.data_plane.init_namespace(context)
        queue = self.pending_by_context[context.key]
        for sample in samples:
            queue.append(RolloutGroupRequest(context=context, sample=sample))

    def _state_for(self, context: TrainingContext) -> RolloutContextState | None:
        pending_groups = len(self.pending_by_context.get(context.key, ()))
        if pending_groups <= 0:
            return None
        partitions = self.data_plane.get_metadata(context)
        capacity = self.staleness_manager.get_rollout_capacity(context, partitions)
        record = self.adapter_registry.get(context)
        open_partitions = [p for p in partitions if p.status == PartitionStatus.OPEN]
        train_ready = [p for p in partitions if p.status == PartitionStatus.TRAIN_READY]
        return RolloutContextState(
            context=context.with_policy_version(record.policy_version, record.adapter_revision),
            pending_groups=pending_groups,
            in_flight_rollouts=record.in_flight_rollouts,
            live_partitions=len([p for p in partitions if p.status != PartitionStatus.CLEARED]),
            open_partitions=len(open_partitions),
            train_ready_partitions=len(train_ready),
            rollout_capacity=capacity.available_groups,
            last_submit_time=self._last_submit_time[context.key],
            submitted_groups=self._submitted_groups[context.key],
            weight=record.weight,
        )

    def pick_next_training_context(self) -> TrainingContext | None:
        states: list[RolloutContextState] = []
        seen: dict[str, TrainingContext] = {}
        for queue in self.pending_by_context.values():
            if not queue:
                continue
            request = queue[0]
            seen[request.context.key] = request.context
        for context in seen.values():
            if len(self.active_tasks) >= self.max_concurrent_groups:
                break
            if not self.adapter_registry.can_accept_rollout(context):
                continue
            if not self.data_plane.check_capacity(context):
                continue
            state = self._state_for(context)
            if state is None or state.rollout_capacity <= 0:
                continue
            states.append(state)
        return self.rollout_policy.pick_next_context(states)

    def _should_protect(self, request: RolloutGroupRequest) -> bool:
        """Return True if this request has been aborted too many times and must not be aborted again."""
        if not self.partial_rollout_config.enabled:
            return False
        return request.abort_count >= self.partial_rollout_config.max_aborted_count

    async def run_one_group(self, request: RolloutGroupRequest) -> RolloutGroupResult:
        context = request.context
        sample = request.sample
        tool_manager = self.tool_manager_factory.create(sample, context)
        trajectory = sample.get('trajectory') or sample
        self.adapter_registry.on_rollout_started(context)
        try:
            rollout_kwargs = {'tool_manager': tool_manager, 'adapter_name': context.adapter_name}
            if context.adapter_revision is not None:
                rollout_kwargs['adapter_path'] = context.adapter_revision
            # Pass partial_state to rollout if resuming an aborted request
            if request.is_resumed and self.partial_rollout_config.enabled:
                rollout_kwargs['partial_state'] = request.partial_state
            result = self.rollout([trajectory], **rollout_kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            # Rollout may signal an abort by returning a RolloutGroupResult directly
            if isinstance(result, RolloutGroupResult):
                if result.status == 'aborted' and self.partial_rollout_config.enabled:
                    record = self.adapter_registry.get(context)
                    record.abort_count += 1
                    return RolloutGroupResult(
                        request=request,
                        status='aborted',
                        partial_state=result.partial_state,
                        error=result.error,
                    )
                trajectories = result.trajectories
            else:
                trajectories = list(result)

            partition_id = self._select_or_create_partition(context)
            meta = self.data_plane.put_rollout_batch(
                context,
                partition_id,
                trajectories,
                ready_groups=1,
                seal=True,
            )
            self.adapter_registry.on_partition_created(context, partition_id)
            self._last_submit_time[context.key] = time.time()
            self._submitted_groups[context.key] += 1
            return RolloutGroupResult(request=request, trajectories=trajectories, status='ok', partition_meta=meta)
        except Exception as exc:
            return RolloutGroupResult(request=request, status='failed', error=str(exc))
        finally:
            self.adapter_registry.on_rollout_finished(context)

    def _select_or_create_partition(self, context: TrainingContext) -> str:
        open_partitions = self.data_plane.list_partitions(context, statuses=[PartitionStatus.OPEN])
        if open_partitions:
            return open_partitions[0].partition_id
        meta = self.data_plane.create_partition(context, target_groups=self.target_groups_per_partition)
        return meta.partition_id

    async def step(self) -> RolloutGroupResult | None:
        context = self.pick_next_training_context()
        if context is None:
            return None
        queue = self.pending_by_context[context.key]
        request = queue.popleft()
        result = await self.run_one_group(request)
        if result.status == 'aborted' and self.partial_rollout_config.enabled:
            # Recycle aborted request: update abort_count, set protected flag, push back to front
            recycled = RolloutGroupRequest(
                context=request.context,
                sample=request.sample,
                partial_state=result.partial_state,
                abort_count=request.abort_count + 1,
                protected=self._should_protect(
                    RolloutGroupRequest(
                        context=request.context, sample=request.sample, abort_count=request.abort_count + 1)),
            )
            queue.appendleft(recycled)
        return result


class RewardWorker:

    def __init__(self, *, data_plane: TransferQueueDataPlane, reward_registry: dict[str, Callable[..., list[float]]]):
        self.data_plane = data_plane
        self.reward_registry = reward_registry

    def run_once(self, context: TrainingContext, *, batch_size: int = 1024) -> PartitionMetadata:
        meta, samples = self.data_plane.claim_reward_batch(context, batch_size)
        reward_fn = self.reward_registry.get(context.reward_type)
        if reward_fn is None:
            raise KeyError(f'unknown reward_type: {context.reward_type}')
        trajectories = [s.get('trajectory', s) for s in samples]
        rewards = list(reward_fn(trajectories, context=context))
        return self.data_plane.append_rewards(context, meta.partition_id, rewards)


class AdvantageWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        advantage_fn: Callable[[list[SampleRecord], TrainingContext], tuple[list[float], list[float]]] | None = None,
    ):
        self.data_plane = data_plane
        self.advantage_fn = advantage_fn or self._default_advantage_fn

    @staticmethod
    def _default_advantage_fn(samples: list[SampleRecord], context: TrainingContext) -> tuple[list[float], list[float]]:
        rewards = [float(sample.get('rewards', sample.get('reward', 0.0))) for sample in samples]
        if not rewards:
            return [], []
        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        return advantages, rewards

    def run_once(self, context: TrainingContext, *, batch_size: int = 1024) -> PartitionMetadata:
        meta, samples = self.data_plane.claim_advantage_batch(context, batch_size)
        advantages, returns = self.advantage_fn(samples, context)
        return self.data_plane.append_advantages(context, meta.partition_id, advantages, returns)


class TrainerScheduler:

    def __init__(self, *, adapter_registry: AdapterRegistry, train_policy: Any | None = None):
        self.adapter_registry = adapter_registry
        self.train_policy = train_policy or PreferCurrentTrainPolicy()

    def next_partition(
        self,
        candidates: list[PartitionMetadata],
        current_context: TrainingContext | None = None,
    ) -> PartitionMetadata | None:
        filtered = []
        for partition in candidates:
            if partition.status != PartitionStatus.TRAIN_READY:
                continue
            if not self.adapter_registry.can_train(partition.context):
                continue
            filtered.append(partition)
        return self.train_policy.pick_next_partition(filtered, current_context)


@dataclass
class TrainerStepResult:
    adapter_revision: str | None = None
    metrics: dict[str, Any] | None = None


class TrainerWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        adapter_registry: AdapterRegistry,
        scheduler: TrainerScheduler,
        train_partition_fn: Callable[[TrainingContext, str, Any], TrainerStepResult | dict[str, Any] | None],
        receive_weights_fn: Callable[[TrainingContext], None] | None = None,
    ):
        self.data_plane = data_plane
        self.adapter_registry = adapter_registry
        self.scheduler = scheduler
        self.train_partition_fn = train_partition_fn
        self.receive_weights_fn = receive_weights_fn
        self.current_context: TrainingContext | None = None

    def run_once(self) -> PartitionMetadata | None:
        partition = self.scheduler.next_partition(
            self.data_plane.list_train_ready_partitions(),
            self.current_context,
        )
        if partition is None:
            return None
        context = partition.context
        self.current_context = context
        self.adapter_registry.on_train_started(context, partition.partition_id)
        self.data_plane.mark_training(context, partition.partition_id)
        dataloader = self.data_plane.build_streaming_dataloader(context, partition.partition_id)
        try:
            result = self.train_partition_fn(context, partition.partition_id, dataloader)
            adapter_revision = None
            if isinstance(result, TrainerStepResult):
                adapter_revision = result.adapter_revision
            elif isinstance(result, dict):
                adapter_revision = result.get('adapter_revision')
            self.data_plane.mark_trained(context, partition.partition_id)
            self.adapter_registry.on_train_finished(context, partition.partition_id)
            self.adapter_registry.on_weight_sync_started(context)
            new_context = self.adapter_registry.on_weight_sync_finished(context, adapter_revision=adapter_revision)
            if self.receive_weights_fn is not None:
                self.receive_weights_fn(new_context)
            self.data_plane.clear_partition(context, partition.partition_id)
            self.adapter_registry.on_partition_cleared(context, partition.partition_id)
            return partition
        except Exception as exc:
            self.adapter_registry.mark_failed(context, str(exc))
            raise
