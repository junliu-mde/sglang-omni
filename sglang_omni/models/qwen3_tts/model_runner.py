# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.types import (
    PRECOMMITTED_PENALTY_CUMULATE_ATTR,
    PrecommittedPenaltyCumulate,
    RequestOutput,
)


@dataclass
class _Qwen3TTSDecodeLaunch:
    """Device snapshots owned by one asynchronous Qwen3-TTS decode step."""

    code_snapshot: torch.Tensor
    feedback_snapshot: torch.Tensor
    feedback_rows: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _LookaheadRepetitionToken:
    """One staged semantic id pending host output finalization."""

    result: Any
    row_index: int


_LOOKAHEAD_REPETITION_TOKEN_ATTR = "_omni_lookahead_repetition_token"
_LOOKAHEAD_REPETITION_SYNC_FALLBACK_ATTR = (
    "_omni_lookahead_repetition_sync_fallback"
)


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

    def _prepare_and_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        is_prefill: bool,
        *,
        is_lookahead: bool = False,
    ) -> Any:
        """Start the semantic-id DtoH copy before Predictor GPU work.

        Qwen3-TTS samples the semantic id before its Predictor pass. Staging it
        here lets the dedicated copy stream transfer the id while the Predictor
        computes codec ids and feedback on the decode stream.
        """

        result = super()._prepare_and_forward(
            forward_batch,
            schedule_batch,
            requests,
            is_prefill,
            is_lookahead=is_lookahead,
        )
        token_ids = result.next_token_ids
        if isinstance(token_ids, torch.Tensor) and token_ids.is_cuda:
            self._stage_token_ids(result, token_ids)
        return result

    def _ensure_next_token_ids(
        self,
        batch_result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        scheduler_output: Any,
    ) -> None:
        """Cover any future path that materializes ids after the post hook."""

        super()._ensure_next_token_ids(
            batch_result,
            forward_batch,
            schedule_batch,
            scheduler_output,
        )
        token_ids = batch_result.next_token_ids
        if isinstance(token_ids, torch.Tensor) and token_ids.is_cuda:
            self._stage_token_ids(batch_result, token_ids)

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        self.model.prepare_decode_buffers(requests)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(
            forward_batch,
            input_embeds,
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)
        self._clear_lookahead_repetition_sync_fallbacks(requests)

    def lookahead_eligible(self, batch: Any) -> bool:
        """Use lookahead when Qwen can commit its sampling state on device."""

        # ``batch.reqs`` contains SGLang's raw ``Req`` objects. Omni request
        # data is attached at admission under the private attribute, while the
        # ``SchedulerRequest.data`` wrapper exists only after launch. A missing
        # value must stay on the synchronous path because lookahead does not
        # preserve the normal logprob collection order.
        for req in batch.reqs:
            data = getattr(req, "_omni_data", None)
            if data is None or getattr(data, "return_logprob", True):
                return False
            if getattr(req, _LOOKAHEAD_REPETITION_SYNC_FALLBACK_ATTR, False):
                return False
        return True

    def commit_lookahead_sampling_state(
        self,
        result: Any,
        schedule_batch: Any,
        scheduler_sampling_info: Any,
        requests: list,
    ) -> None:
        """Commit this step's semantic ids before the next lookahead launch.

        SGLang normally updates penalty state in
        ``ScheduleBatch.prepare_for_decode`` from host ``Req.output_ids``.
        During lookahead that list is one step late. The sampled semantic ids
        already reside on the decode stream, so update the original scheduler
        state there and tell ``OmniScheduler`` to skip exactly one later
        host-history update. This keeps repetition, frequency, presence, and
        minimum-new-token state identical to the synchronous sequence.
        """

        del requests
        if scheduler_sampling_info is None:
            return
        orchestrator = getattr(
            scheduler_sampling_info, "penalizer_orchestrator", None
        )
        if orchestrator is None or not orchestrator.is_required:
            return
        token_ids = result.next_token_ids
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise RuntimeError(
                "Qwen3-TTS lookahead requires one device semantic id per request"
            )
        batch_size = len(schedule_batch.reqs)
        if token_ids.shape[0] < batch_size:
            raise RuntimeError(
                "Qwen3-TTS lookahead semantic-id batch is smaller than the "
                "scheduler batch"
            )
        orchestrator.cumulate_output_tokens(token_ids[:batch_size])
        self._stage_lookahead_repetition_tokens(
            result,
            schedule_batch.reqs,
        )
        precommitted_requests = tuple(
            req
            for req in schedule_batch.reqs
            if self._uses_output_history_penalty(req.sampling_params)
        )
        if not precommitted_requests:
            raise RuntimeError(
                "Qwen3-TTS committed penalty state without a matching sampling "
                "parameter"
            )
        setattr(
            schedule_batch,
            PRECOMMITTED_PENALTY_CUMULATE_ATTR,
            PrecommittedPenaltyCumulate(precommitted_requests),
        )

    @staticmethod
    def _uses_output_history_penalty(sampling_params: Any) -> bool:
        """Match every SGLang penalty that reads the newest output token."""

        return (
            getattr(sampling_params, "repetition_penalty", 1.0) != 1.0
            or getattr(sampling_params, "frequency_penalty", 0.0) != 0.0
            or getattr(sampling_params, "presence_penalty", 0.0) != 0.0
            or getattr(sampling_params, "min_new_tokens", 0) > 0
        )

    @staticmethod
    def _stage_lookahead_repetition_tokens(
        result: Any,
        reqs: list,
    ) -> None:
        """Keep the staged semantic id missing from Qwen's host penalty.

        SGLang's penalizer state is committed on device above. Qwen also has a
        legacy repetition-penalty pass that reads ``Req.output_ids`` directly.
        In a lookahead launch that list is one token behind, so retain the
        already-started pinned DtoH snapshot keyed by the raw request. The next
        launch reads it only after enqueuing its Talker forward, so any wait can
        overlap GPU work. Resolve clears the record after the normal host output
        append catches up.
        """

        if not any(
            req.sampling_params.repetition_penalty != 1.0 for req in reqs
        ):
            return
        if not isinstance(getattr(result, "_host_token_ids", None), torch.Tensor):
            # A CUDA Graph capture cannot enqueue the cross-stream DtoH copy.
            # The next scheduler iteration must drain this step and run one
            # normal decode, where Req.output_ids has caught up. This preserves
            # the legacy host repetition pass without doing capture-unsafe work.
            for req in reqs:
                if req.sampling_params.repetition_penalty != 1.0:
                    setattr(req, _LOOKAHEAD_REPETITION_SYNC_FALLBACK_ATTR, True)
            return
        for row_index, req in enumerate(reqs):
            if req.sampling_params.repetition_penalty != 1.0:
                setattr(
                    req,
                    _LOOKAHEAD_REPETITION_TOKEN_ATTR,
                    _LookaheadRepetitionToken(result, row_index),
                )

    @staticmethod
    def _clear_lookahead_repetition_sync_fallbacks(requests: list) -> None:
        """Re-enable lookahead after the required synchronous decode completes."""

        for sched_req in requests:
            req = sched_req.data.req
            if getattr(req, _LOOKAHEAD_REPETITION_SYNC_FALLBACK_ATTR, False):
                delattr(req, _LOOKAHEAD_REPETITION_SYNC_FALLBACK_ATTR)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        """Apply Qwen's host-history penalty plus a lookahead-only device id."""

        super()._apply_repetition_penalty(logits_output, requests)
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return

        rows: list[int] = []
        token_ids: list[int] = []
        penalties: list[float] = []
        resolved_host_ids: dict[int, torch.Tensor] = {}
        for row_index, sched_req in enumerate(requests):
            req = sched_req.data.req
            pending = getattr(req, _LOOKAHEAD_REPETITION_TOKEN_ATTR, None)
            if pending is None:
                continue
            if not isinstance(pending, _LookaheadRepetitionToken):
                raise RuntimeError(
                    "Qwen3-TTS lookahead repetition-token handoff is invalid"
                )
            result_key = id(pending.result)
            host_token_ids = resolved_host_ids.get(result_key)
            if host_token_ids is None:
                host_token_ids = self._resolve_host_token_ids(pending.result)
                if not isinstance(host_token_ids, torch.Tensor):
                    raise RuntimeError(
                        "Qwen3-TTS lookahead repetition-token handoff has no "
                        "host semantic ids"
                    )
                resolved_host_ids[result_key] = host_token_ids
            if not 0 <= pending.row_index < host_token_ids.shape[0]:
                raise RuntimeError(
                    "Qwen3-TTS lookahead repetition-token row is out of range"
                )
            token_id = int(host_token_ids[pending.row_index])
            if not 0 <= token_id < logits.shape[1]:
                continue
            if token_id in {
                int(token)
                for token in req.output_ids
                if 0 <= int(token) < logits.shape[1]
            }:
                continue
            rows.append(row_index)
            token_ids.append(token_id)
            penalties.append(float(req.sampling_params.repetition_penalty))

        if not rows:
            return
        device = logits.device
        row_ids = torch.tensor(rows, dtype=torch.long, device=device)
        current_token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        penalty_tensor = torch.tensor(
            penalties,
            dtype=torch.float32,
            device=device,
        )
        scores = logits[row_ids, current_token_ids].to(torch.float32)
        scores = torch.where(
            scores > 0,
            scores / penalty_tensor,
            scores * penalty_tensor,
        )
        logits[row_ids, current_token_ids] = scores.to(logits.dtype)

    @staticmethod
    def _clear_lookahead_repetition_tokens(
        result: Any,
        requests: list,
    ) -> None:
        """Discard records only when their own launched step is finalized."""

        for sched_req in requests:
            req = sched_req.data.req
            pending = getattr(req, _LOOKAHEAD_REPETITION_TOKEN_ATTR, None)
            if isinstance(pending, _LookaheadRepetitionToken) and pending.result is result:
                delattr(req, _LOOKAHEAD_REPETITION_TOKEN_ATTR)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._install_semantic_sampling_seeds(forward_batch, requests)
        next_token_ids = super()._sample_next_token_ids(
            logits_output,
            forward_batch,
            schedule_batch,
            requests,
        )
        if isinstance(next_token_ids, torch.Tensor):
            self._mask_last_sampled = next_token_ids
        else:
            self._mask_last_sampled = None
        return next_token_ids

    # ------------------------------------------------------------------
    # Mask-based logit shaping (device-resident, replaces the per-step
    # host index building in the base helpers for this model)
    # ------------------------------------------------------------------

    def _mask_fingerprint(self, requests: list) -> list | None:
        rids = []
        for sched_req in requests:
            rid = getattr(sched_req, "request_id", None)
            epoch = getattr(sched_req.data, "_qwen3_tts_prep_epoch", None)
            if rid is None or epoch is None:
                return None
            rids.append((rid, epoch))
        return rids

    @staticmethod
    def _every_row_grew_by_one(requests: list) -> bool:
        """True when every penalized row's history is exactly one token longer.

        Rows at penalty 1.0 are exempt: their mask bits never reach the logits.
        """
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            if float(req.sampling_params.repetition_penalty) == 1.0:
                continue
            seen_len = getattr(data, "_rep_seen_len", None)
            output_ids = req.output_ids
            if seen_len is None or not output_ids:
                return False
            if len(output_ids) != seen_len + 1:
                return False
        return True

    def _ensure_masks(self, batch_size: int, vocab: int, device: Any) -> None:
        masks = getattr(self, "_shape_masks", None)
        if (
            masks is not None
            and masks[0].shape[0] >= batch_size
            and masks[0].shape[1] == vocab
            and masks[0].device == device
        ):
            return
        rows = max(batch_size, 64)
        self._shape_masks = (
            torch.zeros(rows, vocab, dtype=torch.bool, device=device),
            torch.zeros(rows, vocab, dtype=torch.bool, device=device),
            torch.ones(rows, 1, dtype=torch.float32, device=device),
        )
        self._mask_prep_rids = None

    def _rebuild_masks(self, requests: list, vocab: int, device: Any) -> None:
        rep_mask, sup_mask, pen_col = self._shape_masks
        batch_size = len(requests)
        rep_mask[:batch_size] = False
        sup_mask[:batch_size] = False
        rep_rows: list[int] = []
        rep_toks: list[int] = []
        penalties = [1.0] * batch_size
        sup_rows: list[int] = []
        sup_toks: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            penalty = float(req.sampling_params.repetition_penalty)
            penalties[row_idx] = penalty
            output_ids = req.output_ids
            if penalty != 1.0 and output_ids:
                seen = ModelRunner._rep_penalty_unique_tokens(data, output_ids, vocab)
                rep_rows.extend([row_idx] * len(seen))
                rep_toks.extend(seen)
            suppress_tokens = data.suppress_tokens
            if not suppress_tokens:
                suppress_tokens = getattr(req, "_codec_suppress_tokens", None)
            if suppress_tokens:
                for token_id in suppress_tokens:
                    tok = int(token_id)
                    if 0 <= tok < vocab:
                        sup_rows.append(row_idx)
                        sup_toks.append(tok)
        if rep_rows:
            pairs = torch.tensor(rep_rows + rep_toks, dtype=torch.long, device=device)
            rep_mask[pairs[: len(rep_rows)], pairs[len(rep_rows) :]] = True
        if sup_rows:
            pairs = torch.tensor(sup_rows + sup_toks, dtype=torch.long, device=device)
            sup_mask[pairs[: len(sup_rows)], pairs[len(sup_rows) :]] = True
        pen_col[:batch_size, 0] = torch.tensor(
            penalties, dtype=torch.float32, device=device
        )
        self._mask_rep_active = bool(rep_rows) or any(p != 1.0 for p in penalties)
        self._mask_sup_active = bool(sup_rows)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        batch_size = len(requests)
        vocab = logits.shape[1]
        self._ensure_masks(batch_size, vocab, logits.device)
        rep_mask, sup_mask, pen_col = self._shape_masks
        fingerprint = self._mask_fingerprint(requests)
        last_sampled = getattr(self, "_mask_last_sampled", None)
        if (
            fingerprint is not None
            and fingerprint == getattr(self, "_mask_prep_rids", None)
            and last_sampled is not None
            and last_sampled.shape[0] >= batch_size
            and self._every_row_grew_by_one(requests)
        ):
            # Note: (Jiaxin Deng) unchanged batch, every history exactly one token
            # longer: the only new information is each row's sampled token, so one
            # scatter replaces the full host-side index rebuild. A history that did
            # not grow by one (retract, restart) can need bits cleared, which the
            # scatter cannot do, so those steps rebuild.
            if self._mask_rep_active:
                rows = torch.arange(batch_size, device=logits.device)
                rep_mask[rows, last_sampled[:batch_size].clamp(0, vocab - 1)] = True
                for sched_req in requests:
                    data = sched_req.data
                    output_ids = sched_req.data.req.output_ids
                    if output_ids:
                        ModelRunner._rep_penalty_unique_tokens(data, output_ids, vocab)
        else:
            self._rebuild_masks(requests, vocab, logits.device)
        self._mask_prep_rids = fingerprint
        if self._mask_rep_active:
            pen = pen_col[:batch_size]
            scores = logits.to(torch.float32)
            penalized = torch.where(scores > 0, scores / pen, scores * pen)
            logits.copy_(
                torch.where(rep_mask[:batch_size], penalized, scores).to(logits.dtype)
            )

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        masks = getattr(self, "_shape_masks", None)
        if masks is None or not getattr(self, "_mask_sup_active", False):
            return
        logits.masked_fill_(masks[1][: len(requests)], float("-inf"))

    def _install_semantic_sampling_seeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        forward_batch.sampling_info.sampling_seed = (
            self.model._semantic_sampling_seed_tensor[:batch_size]
        )

    def _collect_codes(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        result._qwen3_tts_has_code_step = False
        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        semantic_positions = self._sample_positions(forward_batch, layer0_codes.device)
        self.model.code_predictor_forward(
            layer0_codes,
            hidden,
            semantic_positions=semantic_positions,
        )
        result._qwen3_tts_has_code_step = True

    def post_decode_launch(
        self,
        result: Any,
        forward_batch: Any,
        requests: list,
    ) -> _Qwen3TTSDecodeLaunch | None:
        """Run Predictor before resolve and publish its feedback for lookahead.

        The next decode launch consumes this step's Predictor feedback before
        the scheduler resolves the semantic id on the host. The Predictor owns
        reusable output buffers, so retain private device snapshots for the
        later resolve/output phase.
        """

        if not requests:
            return None
        if result.next_token_ids is None:
            result.next_token_ids = self._sample_next_token_ids(
                result.logits_output,
                forward_batch,
                None,
                requests,
            )
        token_ids = result.next_token_ids
        if isinstance(token_ids, torch.Tensor) and token_ids.is_cuda:
            self._stage_token_ids(result, token_ids)
        self._collect_codes(result, forward_batch, None, requests)
        if not getattr(result, "_qwen3_tts_has_code_step", False):
            return None

        batch_size = len(requests)
        code_snapshot = self.model._output_codes[:batch_size].detach().clone()
        feedback_snapshot = self.model._output_embeds[:batch_size].detach().clone()
        # Keep the exact row views that enter the request queues. A new view of
        # the same tensor would not compare by identity when an EOS row must be
        # removed during resolve.
        feedback_rows = tuple(
            feedback_snapshot[row_idx] for row_idx in range(batch_size)
        )
        for row_idx, sched_req in enumerate(requests):
            sched_req.data.pending_feedback_queue.append(feedback_rows[row_idx])

        result._qwen3_tts_code_snapshot = code_snapshot
        result._qwen3_tts_feedback_snapshot = feedback_snapshot
        result._qwen3_tts_feedback_rows = feedback_rows
        result._qwen3_tts_feedback_prepublished = True
        return _Qwen3TTSDecodeLaunch(
            code_snapshot=code_snapshot,
            feedback_snapshot=feedback_snapshot,
            feedback_rows=feedback_rows,
        )

    def post_decode_resolve(
        self,
        launch_buf: _Qwen3TTSDecodeLaunch | None,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Restore launch-owned snapshots after the lookahead completion event."""

        del forward_batch, schedule_batch, requests
        if launch_buf is None:
            return
        result._qwen3_tts_code_snapshot = launch_buf.code_snapshot
        result._qwen3_tts_feedback_snapshot = launch_buf.feedback_snapshot
        result._qwen3_tts_feedback_rows = launch_buf.feedback_rows
        result._qwen3_tts_feedback_prepublished = True

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        self._clear_lookahead_repetition_tokens(
            result,
            scheduler_output.requests,
        )
        if not getattr(result, "_qwen3_tts_has_code_step", False):
            return
        result._qwen3_tts_has_code_step = False

        code_snapshot = getattr(result, "_qwen3_tts_code_snapshot", None)
        feedback_snapshot = getattr(result, "_qwen3_tts_feedback_snapshot", None)
        feedback_rows = getattr(result, "_qwen3_tts_feedback_rows", None)
        feedback_prepublished = bool(
            getattr(result, "_qwen3_tts_feedback_prepublished", False)
        )
        has_async_snapshot = code_snapshot is not None and feedback_rows is not None
        if code_snapshot is None:
            code_snapshot = self.model._output_codes
        if feedback_snapshot is None:
            feedback_snapshot = self.model._output_embeds
        eos_id = int(self.model.config.codec_eos_token_id)
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            code_chunk = code_snapshot[row_idx]
            feedback = (
                feedback_rows[row_idx]
                if feedback_rows is not None
                else feedback_snapshot[row_idx]
            )
            is_terminal = req_output.data is None or int(req_output.data) == eos_id
            if is_terminal:
                if feedback_prepublished:
                    self._discard_prepublished_feedback(sched_req.data, feedback)
                continue
            if not has_async_snapshot:
                code_chunk = code_chunk.detach().clone()
                feedback = feedback.detach().clone()
            sched_req.data.output_codes.append(code_chunk)
            if not feedback_prepublished:
                sched_req.data.pending_feedback_queue.append(feedback)

    @staticmethod
    def _discard_prepublished_feedback(data: Any, feedback: torch.Tensor) -> None:
        """Remove a feedback row that was published for an EOS semantic id."""

        queue = getattr(data, "pending_feedback_queue", None)
        if queue is None:
            return
        for index, queued_feedback in enumerate(queue):
            if queued_feedback is feedback:
                del queue[index]
                return

    def _sample_positions(
        self, forward_batch: Any, device: torch.device
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        if is_decode:
            positions = getattr(forward_batch, "positions", None)
            if positions is not None:
                return positions.to(device=device, dtype=torch.long)

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            return (seq_lens.to(device=device, dtype=torch.long) - 1).clamp_min(0)

        positions = getattr(forward_batch, "positions", None)
        if positions is not None:
            return positions.to(device=device, dtype=torch.long)

        raise RuntimeError("Qwen3-TTS subtalker sampling requires semantic positions")

    def _write_feedback_buffers(self, forward_batch: Any, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        decode_feedback_embedding = self.model._decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        row_ids = self._decode_row_ids(batch_size, input_ids)
        rows = []

        for row_idx, sched_req in enumerate(requests):
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=decode_feedback_embedding.weight.device,
                dtype=decode_feedback_embedding.weight.dtype,
            )
            if combined is None:
                token_id = input_ids[row_idx : row_idx + 1].to(
                    device=decode_feedback_embedding.weight.device
                )
                combined = self.model.get_input_embeddings()(token_id).reshape(-1)
            QwenTalkerModelRunner._append_decode_input_history(sched_req.data, combined)
            rows.append(combined)
        with torch.no_grad():
            torch.stack(rows, dim=0, out=decode_feedback_embedding.weight[:batch_size])
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)

    def _decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = getattr(self, "_row_ids_cache", None)
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self._row_ids_cache = cached
        return cached[:batch_size]

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            piece = QwenTalkerModelRunner._projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=req_len,
                device=forward_batch.input_ids.device,
            )
            if piece is None or int(piece.shape[0]) != req_len:
                have = 0 if piece is None else int(piece.shape[0])
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {have} rows, need {req_len}"
                )
            pieces.append(piece)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
