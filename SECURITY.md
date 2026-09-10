# Security policy

## Supported versions

Security fixes are applied to the latest `main` branch and newest tagged release.

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Use GitHub's private security
advisory interface for `dancher00/agentic-cad`, or contact the repository owner
privately through the GitHub profile if advisories are unavailable. Include a
minimal reproduction, affected revision, impact, and suggested mitigation when
known.

## Relevant threat boundaries

Agentic CAD processes untrusted images, videos, camera bundles, masks, YAML, meshes,
and generated CadQuery source. The project:

- validates paths, arrays, transforms, finite values, and mesh topology;
- applies an AST allow-list to generated source;
- executes CAD generation in an isolated subprocess with memory, CPU, and wall
  limits;
- refuses existing output directories and divergent external artifacts;
- hashes external source, weights, captures, and inputs;
- reads provider credentials locally and sends text and photos to the selected API.

The CAD subprocess removes environment variables whose names contain KEY, TOKEN,
SECRET or PASSWORD. This filter is not a complete credential isolation boundary.

These controls reduce risk but do not make the program a hardened multi-tenant
sandbox. Run unknown inputs under an OS/container account without sensitive
filesystem access. Do not expose the CLI directly as a public upload service.

Only fixed HTTPS endpoints in checked-in acquisition scripts should be trusted.
Review upstream model/data terms and code before enabling network downloads.
