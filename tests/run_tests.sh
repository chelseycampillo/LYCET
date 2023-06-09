#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script
# Copyright 2022 Harvard University. All Rights Reserved.
set -e

# list of test cases you want to run
tests=(
    test_DualNumber.py
    test_LYCET_operations.py
    test_ForwardMode.py
    test_node_reverse_mode.py
)


export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# decide what driver to use (depending on arguments given)
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
    driver="${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
    # Assumes the package has been installed and dependencies resolved.  This
    # would be the situation for a customer.  Uses `pytest` for testing.
    shift
    unset PYTHONPATH
    driver="pytest ${@}"
else
    driver="python ${@} -m unittest"
fi

# run the tests
${driver} ${tests[@]}