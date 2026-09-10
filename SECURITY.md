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

The product accepts text and JPEG, PNG or WebP photos, receives generated
CadQuery source from a remote provider, and builds CAD and viewer meshes locally.
It:

- checks input paths, file sizes and image formats;
- applies a geometric AST allow-list to model-generated source;
- executes CAD in a subprocess with memory, CPU and wall-time limits;
- checks that the result is one valid, positive-volume solid;
- refuses existing output directories and records input image hashes;
- reads provider credentials locally and sends text and photos to the selected API.

Optional research commands additionally process video, calibrated camera/depth
bundles, masks, YAML and external weights. They are separate from the product
reconstruction path.

The CAD subprocess removes environment variables whose names contain KEY, TOKEN,
SECRET or PASSWORD. This filter is not a complete credential isolation boundary.

These controls reduce risk but do not make the program a hardened multi-tenant
sandbox. Run unknown inputs under an OS/container account without sensitive
filesystem access. Do not expose the CLI directly as a public upload service.

Only fixed HTTPS endpoints in checked-in acquisition scripts should be trusted.
Review upstream model/data terms and code before enabling network downloads.
