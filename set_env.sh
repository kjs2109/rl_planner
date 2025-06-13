#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ./rl_env/bin/activate
source ./install/setup.bash
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/src
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/ros/rl_planning_simulator/rl_planning_simulator
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/rl_env/lib/python3.10/site-packages
export QT_QPA_PLATFORM_PLUGIN_PATH=$SCRIPT_DIR/rl_env/lib/python3.10/site-packages/cv2/qt/plugins