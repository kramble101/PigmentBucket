"""High-level analysis orchestrator."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cache import FeatureCache
from .clusterer import ClustererConfig, KMeansClusterer
from .features import FeatureExtractionError, FeatureExtractor, FeatureExtractorConfig
from .locations import LocationAssignmentResult, LocationClip, LocationConfig, LocationGrouper
from .sampler import FrameSampler, SamplerConfig, SamplingError
from .types import AnalysisResult, ClipAnalysis, ClipContext, ClipFeatures, ClusterResult, FrameSample
from fractions import Fraction
import cv2

np.random.seed(42)


AUTO_SIM_TARGET = 0.99
AUTO_SIM_STEP = 0.03
AUTO_MIN_DURATION_TARGET = 1.0
AUTO_MIN_DURATION_STEP = 0.5


@dataclass
class AnalyzerConfig:
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    feature: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    cluster: ClustererConfig = field(default_factory=ClustererConfig)
    force_min_k: int = 1
    ignore_cache: bool = False
    cache_dir: Path = field(default_factory=lambda: Path(".cache/features"))
    location_min_duration_sec: float = 3.0
    location_similarity_threshold: float = 0.92
    location_hysteresis: bool = True
    auto_mode: bool = False
    auto_target_k: int = 2
    auto_max_iters: int = 3
    auto_improvement_epsilon: float = 0.03
    auto_silhouette_threshold: float = 0.08
    auto_profiles: Sequence[Dict[str, object]] = field(default_factory=list)
    project_id: Optional[str] = None
    clip_context_hash: Optional[str] = None
    dump_frames: bool = False
    frames_dir: Optional[Path] = None


@dataclass
class AutoAttempt:
    iteration: int
    similarity: float
    min_duration: float
    random_state: int
    chosen_k: int
    unique_clusters: int
    silhouette: Optional[float]
    decision: str


class Analyzer:
    """Coordinates sampling, feature extraction, and clustering."""

    def __init__(self, config: AnalyzerConfig | None = None, logger: logging.Logger | None = None) -> None:
        self._config = config or AnalyzerConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._sampler = FrameSampler(self._config.sampler, self._logger)
        self._feature_extractor = FeatureExtractor(self._config.feature)
        self._clusterer = KMeansClusterer(self._config.cluster)
        self._cache: FeatureCache | None = None
        if self._config.cache_dir:
            self._cache = FeatureCache(self._config.cache_dir)
        self._dump_frames_enabled = bool(self._config.dump_frames and self._config.frames_dir)
        self._frames_dir = self._config.frames_dir if self._dump_frames_enabled else None
        if self._frames_dir:
            self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._dumped_frames_count = 0

    def analyze(self, clips: Sequence[ClipContext]) -> AnalysisResult:
        base_results: Dict[str, ClipAnalysis] = {}
        processed_features: Dict[str, ClipFeatures] = {}
        total_samples = 0
        cache_hits = 0

        for clip in clips:
            self._logger.debug("Processing clip %s", clip.clip_id)

            cached_entry = None
            if self._cache and not self._config.ignore_cache:
                cached_entry = self._cache.load(clip, self._config.sampler, self._config.feature)
            if cached_entry:
                features, frame_count = cached_entry
                processed_features[clip.clip_id] = features
                cache_hits += 1
                total_samples += int(features.metadata.get("frame_count", frame_count))
                base_results[clip.clip_id] = ClipAnalysis(
                    clip_id=clip.clip_id,
                    cluster_id=-1,
                    features=features,
                    samples=[],
                    status="ok",
                )
                continue

            try:
                samples = self._sampler.sample(clip)
                if not samples:
                    raise SamplingError("No frames sampled")
            except SamplingError as error:
                self._logger.warning("Sampling failed for %s: %s", clip.clip_id, error)
                base_results[clip.clip_id] = self._skipped_clip(clip.clip_id, str(error), samples=[])
                continue

            try:
                features = self._feature_extractor.extract(clip.clip_id, samples)
            except FeatureExtractionError as error:
                self._logger.warning("Feature extraction failed for %s: %s", clip.clip_id, error)
                base_results[clip.clip_id] = self._skipped_clip(clip.clip_id, str(error), samples=samples)
                continue

            frame_count = len(samples)
            total_samples += frame_count
            processed_features[clip.clip_id] = features
            base_results[clip.clip_id] = ClipAnalysis(
                clip_id=clip.clip_id,
                cluster_id=-1,
                features=features,
                samples=samples,
                status="ok",
            )

            if self._dump_frames_enabled:
                self._maybe_dump_frames(clip, samples)

            if self._cache and not self._config.ignore_cache:
                try:
                    self._cache.store(clip, self._config.sampler, self._config.feature, features, frame_count)
                except Exception as error:  # pragma: no cover - cache failure is non-fatal
                    self._logger.debug("Cache store failed for %s: %s", clip.clip_id, error)

        processed_count = len(processed_features)
        skipped_count = len(clips) - processed_count

        reference_centroid = self._compute_reference_centroid(processed_features.values())

        auto_initial_similarity = float(self._config.location_similarity_threshold)
        auto_initial_min_duration = float(self._config.location_min_duration_sec)
        auto_initial_source = "default"
        if self._config.auto_mode:
            suggestion = self._select_auto_profile(reference_centroid)
            if suggestion is not None:
                auto_initial_similarity, auto_initial_min_duration = suggestion
                auto_initial_source = "learned"

        history: List[AutoAttempt] = []
        best_result: Optional[AnalysisResult] = None
        best_score = float("-inf")
        selected_result: Optional[AnalysisResult] = None
        selected_attempt: Optional[AutoAttempt] = None

        current_similarity = auto_initial_similarity
        current_min_duration = auto_initial_min_duration
        prev_silhouette: Optional[float] = None
        low_silhouette_retry = False
        max_iters = self._config.auto_max_iters if self._config.auto_mode else 1

        attempt_index = 0
        last_result: Optional[AnalysisResult] = None
        last_meta: Dict[str, object] = {}

        while attempt_index < max_iters:
            attempt_index += 1
            random_state = self._config.cluster.random_state + attempt_index - 1
            result, meta = self._run_attempt(
                clips,
                base_results,
                processed_features,
                cache_hits,
                total_samples,
                processed_count,
                skipped_count,
                current_similarity,
                current_min_duration,
                random_state,
            )
            last_result = result
            last_meta = meta

            silhouette = meta.get("silhouette") if meta else None
            silhouette_value = float(silhouette) if silhouette is not None else None
            chosen_k = int(meta.get("chosen_k", 0) or 0)
            unique_clusters = int(meta.get("unique_clusters", 0) or 0)

            decision, next_similarity, next_min_duration, low_silhouette_retry = self._auto_decision(
                attempt_index,
                max_iters,
                chosen_k,
                unique_clusters,
                silhouette_value,
                prev_silhouette,
                current_similarity,
                current_min_duration,
                low_silhouette_retry,
            )

            attempt_log = AutoAttempt(
                iteration=attempt_index,
                similarity=current_similarity,
                min_duration=current_min_duration,
                random_state=random_state,
                chosen_k=chosen_k,
                unique_clusters=unique_clusters,
                silhouette=silhouette_value,
                decision=decision,
            )
            history.append(attempt_log)

            if self._config.auto_mode:
                self._logger.info(
                    "Auto attempt %d: min_dur=%.2fs sim=%.3f k=%d sil=%s decision=%s",
                    attempt_index,
                    current_min_duration,
                    current_similarity,
                    chosen_k,
                    "n/a" if silhouette_value is None else f"{silhouette_value:.3f}",
                    decision,
                )

            score = chosen_k * 10.0
            if silhouette_value is not None:
                score += silhouette_value
            if score > best_score:
                best_score = score
                best_result = result

            if decision.startswith("accept"):
                selected_result = result
                selected_attempt = attempt_log
                break

            if silhouette_value is not None:
                prev_silhouette = silhouette_value

            current_similarity = next_similarity
            current_min_duration = next_min_duration

        if selected_result is None:
            selected_result = best_result if best_result is not None else last_result
            if history:
                selected_attempt = history[-1]

        if selected_result is None:
            # Fallback to empty result when no clips were processed.
            empty_summary: Dict[str, object] = {
                "processed": processed_count,
                "skipped": skipped_count,
                "clusters": 0,
                "chosen_k": 0,
                "sampled_frames": total_samples,
            }
            return AnalysisResult(clips=[], summary=empty_summary)

        summary = selected_result.summary if isinstance(selected_result.summary, dict) else {}
        summary.setdefault("processed", processed_count)
        summary.setdefault("skipped", skipped_count)
        summary.setdefault("sampled_frames", total_samples)
        if cache_hits:
            summary.setdefault("cache_hits", cache_hits)

        auto_history_payload = [
            {
                "iteration": attempt.iteration,
                "similarity": round(float(attempt.similarity), 4),
                "min_duration": round(float(attempt.min_duration), 3),
                "random_state": attempt.random_state,
                "chosen_k": attempt.chosen_k,
                "unique_clusters": attempt.unique_clusters,
                "silhouette": (
                    round(float(attempt.silhouette), 4)
                    if attempt.silhouette is not None
                    else None
                ),
                "decision": attempt.decision,
            }
            for attempt in history
        ]

        final_similarity = selected_attempt.similarity if selected_attempt else current_similarity
        final_min_duration = selected_attempt.min_duration if selected_attempt else current_min_duration

        is_auto = bool(self._config.auto_mode)
        summary["auto_mode"] = is_auto
        summary["auto_iterations"] = len(history) if history else 1
        summary["auto_similarity_threshold"] = round(float(final_similarity), 4)
        summary["auto_min_duration_sec"] = round(float(final_min_duration), 3)
        summary["auto_history"] = (
            auto_history_payload if is_auto and auto_history_payload else None
        )
        summary["auto_initial_source"] = auto_initial_source
        summary["auto_target_k"] = int(self._config.auto_target_k)
        summary["auto_reference_centroid"] = reference_centroid
        summary["project_id"] = self._config.project_id
        summary["clip_context_hash"] = self._config.clip_context_hash
        if self._dump_frames_enabled:
            summary["dumped_frames_count"] = self._dumped_frames_count
            summary["frames_dir"] = str(self._frames_dir) if self._frames_dir else None
        else:
            summary.setdefault("dumped_frames_count", 0)

        return AnalysisResult(clips=selected_result.clips, summary=summary)

    def _run_attempt(
        self,
        clip_contexts: Sequence[ClipContext],
        base_results: Dict[str, ClipAnalysis],
        processed_features: Dict[str, ClipFeatures],
        cache_hits: int,
        total_samples: int,
        processed_count: int,
        skipped_count: int,
        similarity_threshold: float,
        min_duration_sec: float,
        random_state: int,
    ) -> Tuple[AnalysisResult, Dict[str, object]]:
        attempt_results = {
            clip_id: replace(analysis)
            for clip_id, analysis in base_results.items()
        }
        for analysis in attempt_results.values():
            analysis.cluster_id = -1

        cluster_summary = self._cluster_with_random_state(processed_features.values(), random_state)
        for clip_id, label in cluster_summary.labels.items():
            if clip_id in attempt_results:
                attempt_results[clip_id].cluster_id = int(label)

        unique_clusters = len(set(cluster_summary.labels.values())) if processed_features else 0
        coloring_decision: Dict[str, str] | None = None

        if processed_features and (cluster_summary.chosen_k <= 1 or unique_clusters <= 1):
            force_min_k = max(1, self._config.force_min_k)
            if force_min_k > 1 and len(processed_features) >= force_min_k:
                forced = self._force_split(processed_features, force_min_k)
                if forced:
                    for clip_id, label in forced.items():
                        if clip_id in attempt_results:
                            attempt_results[clip_id].cluster_id = int(label)
                    unique_clusters = len(set(forced.values()))
                    cluster_summary = type(cluster_summary)(
                        labels=forced,
                        chosen_k=unique_clusters,
                        silhouette=None,
                    )
                    coloring_decision = {"coloring": "forced", "reason": "force-min-k"}
                else:
                    coloring_decision = {"coloring": "skipped", "reason": "force-min-k-unavailable"}
            else:
                coloring_decision = {"coloring": "skipped", "reason": "k==1"}

        location_result = self._compute_locations(
            clip_contexts,
            attempt_results,
            min_duration_sec=min_duration_sec,
            similarity_threshold=similarity_threshold,
            hysteresis=self._config.location_hysteresis,
        )
        self._apply_location_metadata(location_result, attempt_results)

        processed_vectors = list(processed_features.values())
        feature_dim = int(processed_vectors[0].vector.shape[0]) if processed_vectors else 0
        silhouette_value = (
            float(cluster_summary.silhouette) if cluster_summary.silhouette is not None else None
        )

        summary: Dict[str, object] = {
            "processed": processed_count,
            "skipped": skipped_count,
            "clusters": unique_clusters,
            "feature_dim": feature_dim,
            "chosen_k": cluster_summary.chosen_k or 0,
            "silhouette": silhouette_value,
            "sampled_frames": total_samples,
        }
        if coloring_decision:
            summary["coloring"] = coloring_decision
        if cache_hits:
            summary["cache_hits"] = cache_hits

        if location_result.location_count:
            summary["locations"] = location_result.location_count
            summary["locations_detail"] = [
                {
                    "location_id": info.location_id,
                    "cluster_id": info.cluster_id,
                    "clip_ids": info.clip_ids,
                    "duration_seconds": info.duration_seconds,
                    "clip_count": info.clip_count,
                    "start_tc": info.start_tc,
                    "end_tc": info.end_tc,
                    "centroid": info.centroid,
                }
                for info in location_result.locations.values()
            ]
            lengths = [info.clip_count for info in location_result.locations.values()]
            durations = [info.duration_seconds for info in location_result.locations.values()]
            summary["locations_stats"] = {
                "clip_count": {
                    "min": int(min(lengths)) if lengths else 0,
                    "median": float(np.median(lengths)) if lengths else 0.0,
                    "max": int(max(lengths)) if lengths else 0,
                },
                "duration_sec": {
                    "min": float(min(durations)) if durations else 0.0,
                    "median": float(np.median(durations)) if durations else 0.0,
                    "max": float(max(durations)) if durations else 0.0,
                },
            }

        ordered_results = [attempt_results[context.clip_id] for context in clip_contexts if context.clip_id in attempt_results]
        meta = {
            "chosen_k": cluster_summary.chosen_k or 0,
            "unique_clusters": unique_clusters,
            "silhouette": silhouette_value,
        }
        return AnalysisResult(clips=ordered_results, summary=summary), meta

    def _cluster_with_random_state(
        self, features: Iterable[ClipFeatures], random_state: int
    ) -> ClusterResult:
        cluster_config = replace(self._config.cluster, random_state=random_state)
        clusterer = KMeansClusterer(cluster_config)
        return clusterer.cluster(features)

    def _maybe_dump_frames(self, context: ClipContext, samples: Sequence[FrameSample]) -> None:
        if not self._frames_dir or not samples:
            return
        timestamps_ms: List[int] = []
        fps = self._infer_clip_fps(context)
        start_frame = self._timecode_to_frames(context.timeline_in, fps) if context.timeline_in else 0
        clip_dir = self._frames_dir
        clip_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            if sample.data is None:
                continue
            ms = int(round(sample.timestamp_seconds * 1000))
            timestamps_ms.append(ms)
            frame_offset = int(round(sample.timestamp_seconds * float(fps))) if fps else 0
            total_frame = (start_frame or 0) + frame_offset
            tc = self._frames_to_timecode(total_frame, fps) if fps else None
            filename = f"{context.clip_id}_t{ms:05d}"
            if tc:
                filename += f"_{tc.replace(':', '')}"
            filename += f"_{sample.frame_index:03d}.jpg"
            frame_path = clip_dir / filename
            bgr = cv2.cvtColor(sample.data, cv2.COLOR_RGB2BGR)
            try:
                cv2.imwrite(str(frame_path), bgr)
                self._dumped_frames_count += 1
            except Exception as error:  # pragma: no cover - best effort tracing
                self._logger.warning("Failed to write frame %s: %s", frame_path, error)

        if timestamps_ms:
            self._logger.info(
                "[SAMPLE] %s frames(ms)=%s saved-> %s",
                context.clip_id,
                ",".join(str(ms) for ms in timestamps_ms[:3]),
                self._frames_dir,
            )

    def _infer_clip_fps(self, context: ClipContext) -> Fraction:
        if context.duration_frames and context.duration_seconds and context.duration_seconds > 0:
            fps_float = max(context.duration_frames / context.duration_seconds, 1.0)
            return Fraction(fps_float).limit_denominator(1000)
        return Fraction(24, 1)

    def _timecode_to_frames(self, timecode: Optional[str], fps: Fraction) -> Optional[int]:
        if not timecode:
            return None
        parts = timecode.split(":")
        if len(parts) != 4:
            return None
        try:
            hours, minutes, seconds, frames = [int(part) for part in parts]
        except ValueError:
            return None
        fps_value = float(fps)
        total_seconds = hours * 3600 + minutes * 60 + seconds
        base_frames = int(round(total_seconds * fps_value))
        return base_frames + frames

    def _frames_to_timecode(self, frames: int, fps: Fraction) -> str:
        fps_value = float(fps)
        if fps_value <= 0:
            fps_value = 24.0
        total_seconds = frames / fps_value
        hours = int(total_seconds // 3600)
        total_seconds -= hours * 3600
        minutes = int(total_seconds // 60)
        total_seconds -= minutes * 60
        seconds = int(total_seconds)
        frame_count = int(round((total_seconds - seconds) * fps_value))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame_count:02d}"

    def _compute_reference_centroid(self, features: Iterable[ClipFeatures]) -> List[float]:
        vectors = [cf.vector for cf in features if cf.vector.size]
        if not vectors:
            return []
        matrix = np.stack(vectors)
        centroid = matrix.mean(axis=0)
        return [float(round(value, 6)) for value in centroid.tolist()]

    def _select_auto_profile(self, reference_centroid: List[float]) -> Optional[Tuple[float, float]]:
        if not reference_centroid or not self._config.auto_profiles:
            return None
        ref_vec = np.array(reference_centroid, dtype=np.float32)
        ref_norm = float(np.linalg.norm(ref_vec))
        if ref_norm == 0.0:
            return None

        best_score = 0.0
        best_pair: Optional[Tuple[float, float]] = None
        for entry in self._config.auto_profiles:
            if not isinstance(entry, dict):
                continue
            centroid = entry.get("centroid")
            similarity = entry.get("similarity")
            min_duration = entry.get("min_duration") or entry.get("min_duration_sec")
            if centroid is None or similarity is None or min_duration is None:
                continue
            candidate_vec = np.array(centroid, dtype=np.float32)
            candidate_norm = float(np.linalg.norm(candidate_vec))
            if candidate_norm == 0.0:
                continue
            score = float(np.dot(ref_vec, candidate_vec) / (ref_norm * candidate_norm))
            if score > 0.95 and score > best_score:
                best_score = score
                best_pair = (float(similarity), float(min_duration))

        return best_pair

    def _at_bounds(
        self,
        new_similarity: float,
        old_similarity: float,
        new_min_duration: float,
        old_min_duration: float,
        tol: float = 1e-4,
    ) -> bool:
        return (
            abs(new_similarity - old_similarity) < tol
            and abs(new_min_duration - old_min_duration) < tol
        )
    def _auto_decision(
        self,
        iteration: int,
        max_iters: int,
        chosen_k: int,
        unique_clusters: int,
        silhouette: Optional[float],
        prev_silhouette: Optional[float],
        current_similarity: float,
        current_min_duration: float,
        low_silhouette_retry: bool,
    ) -> Tuple[str, float, float, bool]:
        if not self._config.auto_mode:
            return "accept", current_similarity, current_min_duration, False

        target_k = max(1, self._config.auto_target_k)
        epsilon = max(0.0, float(self._config.auto_improvement_epsilon))
        threshold = max(0.0, float(self._config.auto_silhouette_threshold))

        has_enough_clusters = chosen_k >= target_k and unique_clusters >= target_k

        # If we don't have enough clusters, push harder towards finer segmentation.
        if not has_enough_clusters:
            if iteration >= max_iters:
                return "accept_max_iters", current_similarity, current_min_duration, False

            next_similarity = min(AUTO_SIM_TARGET, current_similarity + AUTO_SIM_STEP)
            next_min_duration = max(
                AUTO_MIN_DURATION_TARGET, current_min_duration - AUTO_MIN_DURATION_STEP
            )
            if self._at_bounds(next_similarity, current_similarity, next_min_duration, current_min_duration):
                return "accept_bounds", current_similarity, current_min_duration, False
            return "retry_low_k", next_similarity, next_min_duration, False

        # Enough clusters: evaluate silhouette quality and improvement.
        improvement = None
        if silhouette is not None and prev_silhouette is not None:
            improvement = silhouette - prev_silhouette

        if improvement is not None and improvement >= epsilon:
            return "accept_improved", current_similarity, current_min_duration, False

        if silhouette is not None and silhouette >= threshold:
            return "accept_quality", current_similarity, current_min_duration, False

        if iteration >= max_iters:
            return "accept_max_iters", current_similarity, current_min_duration, False

        # Silhouette still low – allow one retry with a gentler tweak.
        next_similarity = min(AUTO_SIM_TARGET, current_similarity + AUTO_SIM_STEP / 2.0)
        next_min_duration = max(
            AUTO_MIN_DURATION_TARGET, current_min_duration - AUTO_MIN_DURATION_STEP / 2.0
        )

        if low_silhouette_retry or self._at_bounds(
            next_similarity, current_similarity, next_min_duration, current_min_duration
        ):
            return "accept_low_silhouette", current_similarity, current_min_duration, False

        return "retry_low_silhouette", next_similarity, next_min_duration, True

    def _skipped_clip(self, clip_id: str, reason: str, samples: Sequence = ()) -> ClipAnalysis:
        empty_vector = np.zeros(0, dtype=np.float32)
        features = ClipFeatures(clip_id=clip_id, vector=empty_vector, metadata={"feature_dim": 0})
        return ClipAnalysis(
            clip_id=clip_id,
            cluster_id=-1,
            features=features,
            samples=list(samples),
            status="skipped",
            skip_reason=reason,
        )

    def _force_split(self, processed: Dict[str, ClipFeatures], target_k: int) -> Dict[str, int]:
        ids = list(processed.keys())
        if len(ids) < target_k or target_k <= 1:
            return {}
        matrix = np.stack([processed[clip_id].vector for clip_id in ids])
        variances = np.var(matrix, axis=0) if matrix.ndim > 1 else np.array([np.var(matrix)])
        axis = int(np.argmax(variances)) if variances.size else 0
        values = matrix[:, axis] if matrix.ndim > 1 else matrix.astype(float)

        if target_k == 2:
            threshold = float(np.median(values))
            labels = np.where(values > threshold, 1, 0)
            if labels.min() == labels.max():
                order = np.argsort(values)
                midpoint = max(1, len(ids) // 2)
                labels = np.zeros(len(ids), dtype=int)
                labels[order[midpoint:]] = 1
        else:
            labels = np.zeros(len(ids), dtype=int)
            order = np.argsort(values)
            groups = np.array_split(order, target_k)
            for index, group in enumerate(groups):
                labels[group] = index

        if len(set(labels)) < min(target_k, len(ids)):
            return {}
        return {clip_id: int(label) for clip_id, label in zip(ids, labels.tolist())}

    # ------------------------------------------------------------------
    def _compute_locations(
        self,
        clip_contexts: Sequence[ClipContext],
        analyses: Dict[str, ClipAnalysis],
        min_duration_sec: float,
        similarity_threshold: float,
        hysteresis: Optional[bool] = None,
    ) -> LocationAssignmentResult:
        location_clips: List[LocationClip] = []
        for context in clip_contexts:
            analysis = analyses.get(context.clip_id)
            if not analysis or analysis.status != "ok" or analysis.cluster_id is None or analysis.cluster_id < 0:
                continue
            vector = analysis.features.vector
            features_list = vector.tolist() if vector.size else None
            location_clips.append(
                LocationClip(
                    clip_id=context.clip_id,
                    cluster_id=int(analysis.cluster_id),
                    duration_seconds=context.duration_seconds or 0.0,
                    features=features_list,
                    timeline_in=context.timeline_in,
                    timeline_out=context.timeline_out,
                )
            )

        config = LocationConfig(
            min_duration_sec=float(min_duration_sec),
            similarity_threshold=float(similarity_threshold),
            hysteresis=self._config.location_hysteresis if hysteresis is None else bool(hysteresis),
        )
        location_result = LocationGrouper(config).assign(location_clips)
        self._update_location_spans(location_result, clip_contexts)
        return location_result

    def _update_location_spans(
        self,
        location_result: LocationAssignmentResult,
        clip_contexts: Sequence[ClipContext],
    ) -> None:
        if not location_result.locations:
            return
        context_map = {ctx.clip_id: ctx for ctx in clip_contexts}
        for info in location_result.locations.values():
            start_frame: Optional[int] = None
            end_frame: Optional[int] = None
            total_duration = 0.0
            fps_hint: Optional[Fraction] = None
            for clip_id in info.clip_ids:
                context = context_map.get(clip_id)
                if context is None:
                    continue
                fps = self._infer_clip_fps(context)
                if fps_hint is None and fps is not None:
                    fps_hint = fps
                start_tc = context.timeline_in
                end_tc = context.timeline_out
                if start_tc:
                    frame_value = self._timecode_to_frames(start_tc, fps or fps_hint)
                    if frame_value is not None:
                        start_frame = frame_value if start_frame is None else min(start_frame, frame_value)
                if end_tc:
                    frame_value = self._timecode_to_frames(end_tc, fps or fps_hint)
                    if frame_value is not None:
                        end_frame = frame_value if end_frame is None else max(end_frame, frame_value)
                if context.duration_seconds:
                    total_duration += float(context.duration_seconds)

            if fps_hint is None:
                fps_hint = Fraction(24, 1)
            if start_frame is not None:
                info.start_tc = self._frames_to_timecode(start_frame, fps_hint)
            if end_frame is not None:
                info.end_tc = self._frames_to_timecode(end_frame, fps_hint)
            if start_frame is not None and end_frame is not None and end_frame >= start_frame:
                info.duration_seconds = float((end_frame - start_frame) / float(fps_hint))
            elif info.duration_seconds is None:
                info.duration_seconds = total_duration if total_duration > 0 else None

    def _apply_location_metadata(
        self,
        location_result: LocationAssignmentResult,
        analyses: Dict[str, ClipAnalysis],
    ) -> None:
        for clip_id, analysis in analyses.items():
            loc_id = location_result.assignments.get(clip_id)
            analysis.location_id = loc_id if isinstance(loc_id, int) else None
            if analysis.location_id is None:
                continue
            info = location_result.locations.get(analysis.location_id)
            if not info:
                continue
            analysis.location_clip_count = info.clip_count
            analysis.location_duration = info.duration_seconds
            analysis.location_span_start = info.start_tc
            analysis.location_span_end = info.end_tc
            analysis.location_centroid = info.centroid
