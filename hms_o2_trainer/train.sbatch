#!/usr/bin/env bash

[ -e "$HOT_SETUP_ENV" ] && source $HOT_SETUP_ENV

(
  set -x
  echo $@
  cat $1
) 1>&2

exec $@
