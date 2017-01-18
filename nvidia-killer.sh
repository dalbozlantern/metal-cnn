#!/usr/bin/env bash
kill -9 $(nvidia-smi | awk '$2=="Processes:" {p=1} p && $2 == 0 && $3 > 0 {print $3}')