# NOTE (Dual-Space + Parity + Delta Expansion)
 This run uses a dual-space MLP with:
 - positive / negative feature decomposition (ReLU^2)
 - soft parity modulation (low-rank diagonal)
 - local directional expansion via delta = (pos - neg)

 Key idea:
 Each forward pass approximates a local semantic field instead of a single point update.
 The model does not only learn a direction, but also the structure between
 hypothesis (pos) and counter-hypothesis (neg).

 Observed behavior:
 - early phase: directional bias emerges (u_mean drift)
 - mid phase: partial rebalancing
 - later phase: dynamic asymmetry shifts (pos ↔ neg dominance)

# Interpretation:
The dual-space does not collapse into symmetry or pure drift,
but forms an adaptive directional field.

This validates that the negative space is not just passive,
but contributes to meaningful directional structure.