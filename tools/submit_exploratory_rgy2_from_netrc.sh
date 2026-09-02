#!/bin/bash
# Load the HSG-local W&B credential without printing or persisting it, then
# invoke the immutable exploratory launcher.
set -euo pipefail
umask 077

if (( $# != 2 )) || [[ "$1" != "--submit" ]]; then
  printf 'Usage: %s --submit /absolute/path/to/launch_exploratory_rgy2_pair.sh\n' "${0##*/}" >&2
  exit 2
fi
readonly launcher="$2"
readonly netrc_path="/home/jalbericiola/.netrc"
[[ "${launcher}" == /* && -f "${launcher}" && ! -L "${launcher}" ]]
[[ -f "${netrc_path}" && ! -L "${netrc_path}" ]]

wandb_key="$(
  /cm/local/apps/python3/bin/python3 -I -B - "${netrc_path}" <<'PY'
import netrc
import sys

entries = netrc.netrc(sys.argv[1])
auth = entries.authenticators("api.wandb.ai") or entries.authenticators("wandb.ai")
if auth is None or not auth[2]:
    raise SystemExit("W&B credential is absent from netrc")
sys.stdout.write(auth[2])
PY
)"
[[ -n "${wandb_key}" && "${wandb_key}" != *[[:space:]]* && ${#wandb_key} -ge 20 ]]
exec 9<<<"${wandb_key}"
unset wandb_key

exec /usr/bin/env -i \
  PATH=/cm/local/apps/python3/bin:/cm/local/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin \
  HOME=/home/jalbericiola USER=jalbericiola LOGNAME=jalbericiola \
  SHELL=/bin/bash LANG=C LC_ALL=C \
  SLURM_CONF=/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf \
  /bin/bash --noprofile --norc -c '
    set -euo pipefail
    IFS= read -r WANDB_API_KEY <&9
    exec 9<&-
    [[ -n "${WANDB_API_KEY}" && "${WANDB_API_KEY}" != *[[:space:]]* && ${#WANDB_API_KEY} -ge 20 ]]
    export WANDB_API_KEY
    exec /bin/bash --noprofile --norc "$1" --submit
  ' _ "${launcher}"
