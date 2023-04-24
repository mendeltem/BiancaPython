#!/bin/bash

# Function to display help message
show_help() {
  echo "Usage: run_bianca_sh.sh [OPTIONS]"
  echo ""
  echo "This script runs BIANCA shell with specified command line arguments."
  echo ""
  echo "Options:"
  echo "  -image=<value>          Path to the input image in NIfTI format. Required."
  echo "  -mni=<value>            Path to the MNI template image in NIfTI format. Required."
  echo "  -masterfile=<value>     Path to the masterfile containing configuration for BIANCA. Required."
  echo "  -output=<value>         Path to the output directory for BIANCA results. Required."
  echo "  -h, -help               Show this help message and exit."
}

BIANCA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#log
echo "current directory:"
echo $BIANCA_DIR

export BIANCA_DIR

# Check if help option is provided
if [[ "$#" -eq 0 ]]; then
  show_help
  exit 0
fi


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -image=*|--image=*) # Capture -image=<value> or --image=<value>
            IMAGE="${key#*=}"
            shift # past argument=value
            ;;
        -mni=*|--mni=*) # Capture -mni=<value> or --mni=<value>
            MNI="${key#*=}"
            shift # past argument=value
            ;;
        -masterfile=*|--masterfile=*) # Capture -masterfile=<value> or --masterfile=<value>
            MASTERFILE="${key#*=}"
            shift # past argument=value
            ;; 
        -output=*|--output=*) # Capture -output=<value> or --output=<value>
            OUTPUT="${key#*=}"
            shift # past argument=value
            ;;
        -h|--help|-help) # Show help message
        show_help
        exit 0
            ;;
        *)
            # Unknown option
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done


# Check if image path is provided
if [ -z "$IMAGE" ]; then
    echo "Image path not provided. Usage: run_bianca_sh.sh -image=/path/to/image.nii"
    exit 1
fi


# Check if image path is provided
if [ -z "$MNI" ]; then
    echo "Image path not provided. Usage: run_bianca_sh.sh -image=/path/to/image.nii"
    exit 1
fi


# Check if image path is provided
if [ -z "$MASTERFILE" ]; then
    echo "Image path not provided. Usage: run_bianca_sh.sh -image=/path/to/masterfile.txt"
    exit 1
fi


# Check if image path is provided
if [ -z "$OUTPUT" ]; then
    echo "Image path not provided. Usage: run_bianca_sh.sh -image=/path/to/image.nii"
    exit 1
fi


# Change directory to the project directory
cd "$BIANCA_DIR/"

python scripts/check_requirements.py requirements.txt
if [ $? -eq 1 ]
then
    echo Installing missing packages...
    pip install -r requirements.txt
fi
#python -m bianca_shell  $@ -image $1 


#python -m bianca_shell -image $IMAGE

python -m bianca_shell $@  -image="$IMAGE" -mni="$MNI" -masterfile="$MASTERFILE" -output="$OUTPUT"

#read -p "Press any key to continue..."