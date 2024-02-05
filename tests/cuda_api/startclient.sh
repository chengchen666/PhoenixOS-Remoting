#!/bin/bash

LD_PRELOAD=../../target/debug/libclient.so:$LD_PRELOAD $1
