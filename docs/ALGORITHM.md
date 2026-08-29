# Algorithm overview

General Model Registration estimates a rigid 4×4 transformation from a moving triangle mesh to a fixed target mesh. Coordinates are interpreted in millimetres.

## Processing stages

1. Validate and clean both triangle meshes.
2. Sample deterministic, area-weighted surface point clouds.
3. Compute downsampled normals and FPFH features.
4. Generate global candidates using fast global registration and repeated RANSAC.
5. Rank candidates and refine the best candidates with multi-scale robust point-to-plane ICP.
6. Estimate registration quality from bidirectional surface coverage, residuals, normal consistency, spatial coverage and candidate agreement.
7. Generate a high-precision point-to-surface ICP candidate.
8. Accept the high-precision candidate only when coverage, improvement, observability and displacement gates all pass; otherwise retain the multi-scale result.
9. Compute signed closest-surface deviations and export geometry, transformation and quality metadata.

## Determinism

Surface sampling uses local NumPy random generators with fixed seeds. The same inputs and configuration should therefore produce stable samples and candidate evaluation on the same supported software stack. Floating-point and third-party implementation differences can still produce small platform-dependent changes.

## Important limitations

- The transform contains rotation and translation only.
- Symmetric or repetitive geometry may admit multiple plausible poses.
- Low overlap or an insufficiently distributed common surface can make the result underconstrained.
- Signed deviation depends on the target mesh's triangle-normal orientation.
- Confidence metrics describe geometric evidence and are not an accuracy guarantee.
