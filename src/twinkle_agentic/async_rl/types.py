# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format import Trajectory


class PartitionStatus(StrEnum):
    OPEN = 'OPEN'
    ROLLOUT_DONE = 'ROLLOUT_DONE'
    REWARD_DONE = 'REWARD_DONE'
    TRAIN_READY = 'TRAIN_READY'
    TRAINING = 'TRAINING'
    TRAIN_DONE = 'TRAIN_DONE'
    CLEARED = 'CLEARED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class AdapterState(StrEnum):
    LOADING = 'LOADING'
    ACTIVE = 'ACTIVE'
    DRAINING = 'DRAINING'
    CANCELLED = 'CANCELLED'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class TrainingContext:
    tenant_id: str
    training_run_id: str
    base_model_id: str
    adapter_name: str
    adapter_revision: str | None = None
    policy_version: int = 0
    env_type: str = 'tool_calling'
    tool_profile: str = 'default'
    reward_type: str = 'default'
    loss_type: str = 'default'
    algorithm: str = 'grpo'

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def partition_id(self, train_id: int | str) -> str:
        suffix = train_id if isinstance(train_id, str) and train_id.startswith('train_') else f'train_{train_id}'
        return f'{self.key}/{suffix}'

    def with_policy_version(self, policy_version: int, adapter_revision: str | None = None) -> TrainingContext:
        return TrainingContext(
            tenant_id=self.tenant_id,
            training_run_id=self.training_run_id,
            base_model_id=self.base_model_id,
            adapter_name=self.adapter_name,
            adapter_revision=self.adapter_revision if adapter_revision is None else adapter_revision,
            policy_version=policy_version,
            env_type=self.env_type,
            tool_profile=self.tool_profile,
            reward_type=self.reward_type,
            loss_type=self.loss_type,
            algorithm=self.algorithm,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            'tenant_id': self.tenant_id,
            'training_run_id': self.training_run_id,
            'base_model_id': self.base_model_id,
            'adapter_name': self.adapter_name,
            'adapter_revision': self.adapter_revision,
            'policy_version': self.policy_version,
            'env_type': self.env_type,
            'tool_profile': self.tool_profile,
            'reward_type': self.reward_type,
            'loss_type': self.loss_type,
            'algorithm': self.algorithm,
        }

    def validate_metadata(self, metadata: dict[str, Any], *, strict_policy_version: bool = True) -> None:
        expected = self.metadata()
        for key, expected_value in expected.items():
            if key == 'adapter_revision':
                continue
            if key == 'policy_version' and not strict_policy_version:
                continue
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f'context metadata mismatch for {key}: expected {expected_value!r}, got {actual_value!r}')


@dataclass
class PartitionMetadata:
    context: TrainingContext
    partition_id: str
    policy_version: int
    target_groups: int = 0
    ready_groups: int = 0
    status: PartitionStatus = PartitionStatus.OPEN
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    owner_worker_id: str | None = None
    lease_deadline: float | None = None
    num_rows: int = 0

    @property
    def logical_train_id(self) -> str:
        return self.partition_id.rsplit('/', 1)[-1]

    def touch(self) -> None:
        self.updated_at = time.time()

    def tag(self) -> dict[str, Any]:
        tag = self.context.metadata()
        tag.update({
            'partition_id': self.partition_id,
            'policy_version': self.policy_version,
            'target_groups': self.target_groups,
            'ready_groups': self.ready_groups,
            'status': self.status.value,
            'num_rows': self.num_rows,
        })
        return tag


@dataclass
class AdapterRecord:
    tenant_id: str
    training_run_id: str
    adapter_name: str
    base_model_id: str
    state: AdapterState = AdapterState.LOADING
    policy_version: int = 0
    adapter_revision: str | None = None
    train_slot_name: str | None = None
    rollout_slot_name: str | None = None
    live_partitions: set[str] = field(default_factory=set)
    in_flight_rollouts: int = 0
    training_partition: str | None = None
    sync_in_progress: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_error: str | None = None
    abort_count: int = 0
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass(frozen=True)
class RolloutCapacity:
    available_groups: int
    action: str = 'submit'
    reason: str = ''
    sleep_seconds: float = 0.0

    @property
    def can_submit(self) -> bool:
        return self.available_groups > 0 and self.action == 'submit'


@dataclass
class RolloutContextState:
    context: TrainingContext
    pending_groups: int
    in_flight_rollouts: int
    live_partitions: int
    open_partitions: int
    train_ready_partitions: int
    rollout_capacity: int
    last_submit_time: float = 0.0
    submitted_groups: int = 0
    weight: float = 1.0

    @property
    def context_key(self) -> str:
        return self.context.key


SampleRecord = Dict[str, Any]
RewardFn = Any
AdvantageFn = Any
TrainResult = Dict[str, Any]
ContextKey = Tuple[str, str, str]


@dataclass
class PartialRolloutConfig:
    """Controls interrupt-and-resume behavior for in-flight rollouts."""
    enabled: bool = False
    max_aborted_count: int = 3
    mask_offpolicy_tokens: bool = True


@dataclass
class RolloutGroupRequest:
    """A unit of rollout work that can be aborted and resumed.

    `partial_state` carries already-generated tokens/messages when a request
    is recycled after an abort. `abort_count` tracks how many times this
    request has been interrupted; once it reaches the configured limit the
    request becomes `protected` and will not be aborted again (anti-starvation).
    """
    context: TrainingContext
    sample: SampleRecord
    partial_state: dict[str, Any] | None = None
    abort_count: int = 0
    protected: bool = False

    @property
    def is_resumed(self) -> bool:
        return self.partial_state is not None


@dataclass
class RolloutGroupResult:
    """Outcome of run_one_group; `status` distinguishes completion from abort."""
    request: RolloutGroupRequest
    trajectories: list[Trajectory] = field(default_factory=list)
    status: str = 'ok'  # ok / aborted / failed
    partition_meta: PartitionMetadata | None = None
    partial_state: dict[str, Any] | None = None
    error: str | None = None
