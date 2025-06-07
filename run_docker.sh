#!/bin/bash

# Prompt functions
choose_network() {
    local networks=("newcrfs" "pixelformer" "meter")
    PS3=$'\e[1;37mEnter your choice here: \e[0;m'
    echo -e "\e[1;36m--- Network ---\e[0m" >&2
    select network in "${networks[@]}"; do
        [[ -n $network ]] && echo "$network" && break
    done
}

choose_size() {
    local sizes=("tiny" "base" "large")
    PS3=$'\e[1;37mEnter your choice here: \e[0;m'
    echo -e "\e[1;36m--- Network size ---\e[0m" >&2
    select size in "${sizes[@]}"; do
        [[ -n $size ]] && echo "$size" && break
    done
}

choose_dataset() {
    local datasets=("nyu" "kitti")
    PS3=$'\e[1;37mEnter your choice here: \e[0;m'
    echo -e "\e[1;36m--- Dataset ---\e[0m" >&2
    select dataset in "${datasets[@]}"; do
        [[ -n $dataset ]] && echo "$dataset" && break
    done
}

choose_mode() {
    local modes=("train" "test")
    PS3=$'\e[1;37mEnter your choice here: \e[0;m'
    echo -e "\e[1;36m--- Mode ---\e[0m" >&2
    select mode in "${modes[@]}"; do
        [[ -n $mode ]] && echo "$mode" && break
    done
}

choose_optimization() {
    local opts=("none" "meta" "pyra" "moh")
    PS3=$'\e[1;37mEnter your choice here: \e[0;m'
    echo -e "\e[1;36m--- Optimization ---\e[0m" >&2
    select opt in "${opts[@]}"; do
        [[ -n $opt ]] && echo "$opt" && break
    done
}

choose_opt_location() {
    local locs=("encoder" "decoder" "full")
    PS3=$'\e[1;37mEnter your choice here: \e[0;m'
    echo -e "\e[1;36m--- Optimization location ---\e[0m" >&2
    select loc in "${locs[@]}"; do
        [[ -n $loc ]] && echo "$loc" && break
    done
}

run_create_arguments_file() {
    echo -e "\e[1;33mCreating arguments file...\e[0m"
    python3 ./create_arguments_file.py "$MAIN_SCRIPT" "$CONFIG_FILE"
    if [[ $? -ne 0 ]]; then
        echo -e "\e[1;31mFailed to create arguments file. Exiting.\e[0m"
        exit 1
    fi
    echo -e "\e[1;32mArguments file created successfully.\e[0m"
}

confirm_and_run() {
    echo -ne "\e[1;33mDo you want to proceed? [y/n]: \e[0m" && read -r choice
    case "$choice" in
        "yes"|"y"|"Y"|"Yes"|"YES")
            echo -e "\e[1;32mProceeding with Docker execution...\e[0m"
            podman run \
                -v "$PROJECT_DIR:/work/project" \
                -v "$DATA_DIR:/work/data" \
                --device nvidia.com/gpu=all \
                --ipc host \
                -u "$(id -u):$(id -g)" \
                "$DOCKER_IMAGE" \
                /usr/bin/python3 "$MAIN_SCRIPT" "$CONFIG_FILE"
            ;;
        "no"|"n"|"N"|"No"|"NO")
            echo -e "\e[1;32mExiting the script. Goodbye!\e[0m"
            exit 0
            ;;
        *)
            echo -e "\e[1;31mInvalid input. Please type 'yes' or 'no'.\e[0m"
            confirm_and_run
            ;;
    esac
}

# Network selection
NETWORK=$(choose_network)
SIZE=$(choose_size)
DATASET=$(choose_dataset)
MODE=$(choose_mode)
OPTIMIZATION=$(choose_optimization)

# Selezione posizione ottimizzazione solo se non è "none"
if [[ $OPTIMIZATION != "none" && $NETWORK != "meter" ]]; then
    OPT_LOCATION=$(choose_opt_location)
elif [[ $NETWORK == "meter" ]]; then
    # Per METER, impostiamo sempre l'ottimizzazione su "full"
    OPT_LOCATION="full"
else
    OPT_LOCATION="none"
fi

# Configurazione dinamica del percorso
if [[ $NETWORK == "newcrfs" ]]; then
    BASE_CONFIG="/work/project/NeWCRFs/config"
    MAIN_SCRIPT="/work/project/main_newcrfs.py"
elif [[ $NETWORK == "pixelformer" ]]; then
    BASE_CONFIG="/work/project/PixelFormer/config"
    MAIN_SCRIPT="/work/project/main_pxf.py"
elif [[ $NETWORK == "meter" ]]; then
    BASE_CONFIG="/work/project/METER/config"
    MAIN_SCRIPT="/work/project/main_meter.py"
fi

# Combinazioni configurazioni
CONFIG_FILE=""
if [[ $NETWORK == "newcrfs" ]]; then
    # Configurazioni NewCRFs
    if [[ $MODE == "train" && $OPTIMIZATION == "none" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "none" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_full_${SIZE}_${DATASET}.txt"
    fi
elif [[ $NETWORK == "pixelformer" ]]; then
    # Configurazioni PixelFormer (stessa logica di NewCRFs, con SIZE integrata)
    if [[ $MODE == "train" && $OPTIMIZATION == "none" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "none" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_full_${SIZE}_${DATASET}.txt"
    fi
elif [[ $NETWORK == "meter" ]]; then
    # Configurazioni METER (stessa logica di NewCRFs, con SIZE integrata)
    if [[ $MODE == "train" && $OPTIMIZATION == "none" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "none" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_meta_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_pyra_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "train" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_train_moh_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "meta" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_meta_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "pyra" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_pyra_full_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "encoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_encoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "decoder" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_decoder_${SIZE}_${DATASET}.txt"
    elif [[ $MODE == "test" && $OPTIMIZATION == "moh" && $OPT_LOCATION == "full" ]]; then
        CONFIG_FILE="$BASE_CONFIG/arguments_test_moh_full_${SIZE}_${DATASET}.txt"
    fi
fi
# Verifica configurazione
if [[ -z $CONFIG_FILE ]]; then
    echo "Combinazione non supportata" >&2
    exit 1
fi

# Percorsi base
PROJECT_DIR="/home/schiavella/thesis_claudio_lorenzo/"
DATA_DIR="/mnt/ssd1/schiavella/"
DOCKER_IMAGE="docker.io/claudioschi21/thesis_alcor_cuda11.8:latest"

echo -e "\e[1;94m********************************* Summary ********************************* \e[0m"
echo -e "\e[1;36mNetwork:\e[0m $NETWORK"
echo -e "\e[1;36mSize:\e[0m $SIZE"
echo -e "\e[1;36mDataset:\e[0m $DATASET"
echo -e "\e[1;36mMode:\e[0m $MODE"
echo -e "\e[1;36mOptimization:\e[0m $OPTIMIZATION"
echo -e "\e[1;36mOptimization location:\e[0m $OPT_LOCATION"
echo -e "\e[1;36mScript:\e[0m $MAIN_SCRIPT"
echo -e "\e[1;36mConfigurations file:\e[0m $CONFIG_FILE"
echo -e "\e[1;36mProject directory (work/project/):\e[0m $PROJECT_DIR"
echo -e "\e[1;36mData directory (work/data/):\e[0m $DATA_DIR"
echo -e "\e[1;36mDocker image:\e[0m $DOCKER_IMAGE"
echo -e "\e[1;94m*************************************************************************** \e[0m"

run_create_arguments_file
confirm_and_run