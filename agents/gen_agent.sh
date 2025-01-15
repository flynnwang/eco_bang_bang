#!/bin/bash
set -e

# Check if exactly one argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <name> <filepath>"
    exit 1
fi

# Assign the argument to a variable
AGENT_NAME="$1"
WEIGHTS_PATH="$2"

AGENT_TEMPLATE=agent_template

rm -rf ${AGENT_NAME}
cp -r ${AGENT_TEMPLATE} ${AGENT_NAME}

cp ../ebb/env/const.py ${AGENT_NAME}/ecobangbang/env/
cp ../ebb/env/luxenv.py ${AGENT_NAME}/ecobangbang/env/
cp ../ebb/model.py ${AGENT_NAME}/ecobangbang/
cp ${WEIGHTS_PATH} ${AGENT_NAME}/ecobangbang

# Update weights file name
WEIGHTS_NAME=$(basename "$WEIGHTS_PATH")
sed -i "s/WEIGHTS_FILE_NAME/$WEIGHTS_NAME/g" "${AGENT_NAME}/ecobangbang/agent.py"

echo "OK! agent: ${AGENT_NAME} created with weights: ${WEIGHTS_NAME}"
