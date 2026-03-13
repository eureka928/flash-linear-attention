from .chunk import chunk_quasar
from .fused_recurrent import fused_recurrent_quasar

# Preserve the upstream gated-delta-product API names expected by FLA.
chunk_gated_delta_product = chunk_quasar
fused_recurrent_gated_delta_product = fused_recurrent_quasar

__all__ = [
    "chunk_gated_delta_product",
    "chunk_quasar",
    "fused_recurrent_gated_delta_product",
    "fused_recurrent_quasar",
]
