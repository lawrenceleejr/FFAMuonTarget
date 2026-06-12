#!/bin/bash
# Run g4bl (G4beamline 3.08) inside the project docker image, headless.
# Usage: scripts/g4bl-docker.sh input.g4bl [param=value ...]
# The current directory is mounted at /work and used as the working dir,
# so write all output files relative to where you invoke this script.
#
# G4beamline locates its Geant4 datasets via the file $G4BL_DIR/.data, which
# must contain the path of the directory holding G4EMLOW8.0 etc.  The image
# ships the datasets with its Geant4 install but no .data file, so we mount
# one in (otherwise g4bl pops up the g4bldata downloader GUI and hangs).
set -e

IMAGE=${G4BL_IMAGE:-ghcr.io/lawrenceleejr/g4beamline:sha-7a8bfe1}
G4DATA=/usr/local/share/geant4/install/4.11.0.2/share/Geant4-11.0.2/data

# mount the repository root and run in the corresponding subdirectory, so
# parallel jobs can run in their own working directories (runs/jobN/)
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")
REL=$(realpath --relative-to="$ROOT" "$PWD")

DOTDATA=$(mktemp)
trap 'rm -f "$DOTDATA"' EXIT
echo "$G4DATA" > "$DOTDATA"

docker run --rm -i \
  -v "$ROOT":/work -w "/work/$REL" \
  -v "$DOTDATA":/opt/g4beamline/build/.data:ro \
  -e QT_QPA_PLATFORM=offscreen \
  "$IMAGE" /opt/g4beamline/build/bin/g4bl "$@"
