# Design

The architecture and implementation gates are specified in `PLAN.md`. Phase A
establishes typed backend boundaries and an end-to-end stub path. The stub path
is deliberately labelled at every output and does not stand in for DA3,
canonicalization, cadrille, or benchmark evidence.

Generated CadQuery is checked by an AST allow-list and executed in a separate
process with address-space and CPU limits, a wall timeout, and a temporary
working directory. This is designed for non-adversarial model output rather than
hostile arbitrary Python.

