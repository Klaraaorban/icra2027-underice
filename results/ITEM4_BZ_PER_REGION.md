# Item 4 — per-region B/Z vs. geometric outcome

Two distinct analyses, since PS117-29 never had sonar supervision applied
(the trust gate correctly disabled it) — there's no "improvement from sonar"
to measure there, only raw geometric error. PS117-39 has both a sonar and
non-sonar twin, so the literal "does B/Z predict whether the fix works"
question is answerable only there.

## PS117-29 (vanilla model, no sonar supervision ever applied): B/Z vs. raw residual

20-frame sliding windows (stride 10) along the track, using the vanilla
COLMAP model's own rendered depth vs. sonar. B/Z computed from perpendicular
camera-baseline spread over mean sonar range, per window.

**Spearman rho(B/Z, mean |residual|) = 0.083, p=0.729 — not significant.**

B/Z swings enormously within this single sequence (0.55 at the start, up to
24.6 near the close-approach at t=971s — Z shrinks toward zero as the ROV
closes on the ice while B, a pure camera-geometry quantity, stays roughly
fixed), but residual doesn't track it. Honest reading: for this model and
this scene, camera-geometry B/Z alone does not predict where the
reconstruction is more or less accurate — the dominant error source here
(established earlier) is COLMAP's global scale corruption under weak
texture, which is not a purely-geometric quantity B/Z captures.

## PS117-39 (sonar vs. non-sonar twin models): B/Z vs. improvement from sonar supervision

(First attempt hit a resource-contention crash (RAM exhaustion) from running
concurrently with the dense-checkpoint training job for item 2 — a real
infrastructure failure, not a result, logged per rule 5. Retried successfully
once the GPU was free.)

28 windows (20-frame, stride 10) covering the full sequence. **Sonar
supervision improved every single window** — `resid_sonar < resid_nosonar`
in all 28/28 cases, a universal, consistent effect across the whole scene,
not concentrated in any particular region.

**Spearman rho(B/Z, improvement magnitude) = -0.262, p=0.178, n=28 — not
significant.** B/Z does not explain *how much* sonar supervision helps in a
given window, even though it clearly helps everywhere. Honest reading: on
PS117-39, sonar supervision's benefit doesn't scale with camera-geometry
favorability the way the original hypothesis expected — it's closer to a
uniform correction than a geometry-gated one. This is a different, and in
some ways more useful, finding than a clean B/Z-predicts-improvement result
would have been: it suggests sonar supervision is broadly robust on this
scene rather than only helping in geometrically-favorable sub-regions.
