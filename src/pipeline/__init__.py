"""Analysis pipeline components for Pigment Bucket."""

from .analyzer import Analyzer, AnalyzerConfig
from .clusterer import Clusterer, ClustererConfig, KMeansClusterer
from .features import FeatureExtractor, FeatureExtractorConfig
from .locations import LocationAssignmentResult, LocationClip, LocationConfig, LocationGrouper
from .sampler import FrameSampler, Sampler, SamplerConfig
from .types import (
    AnalysisResult,
    ClipAnalysis,
    ClipContext,
    ClipFeatures,
    ClusterResult,
    FrameSample,
)

__all__ = [
    "Analyzer",
    "AnalyzerConfig",
    "Clusterer",
    "ClustererConfig",
    "KMeansClusterer",
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "LocationAssignmentResult",
    "LocationClip",
    "LocationConfig",
    "LocationGrouper",
    "Sampler",
    "FrameSampler",
    "SamplerConfig",
    "AnalysisResult",
    "ClipAnalysis",
    "ClipContext",
    "ClipFeatures",
    "ClusterResult",
    "FrameSample",
]
