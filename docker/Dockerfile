FROM python:3.11-slim-bookworm AS base

ARG UID
ARG GID
ARG UNAME

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    gnupg \
    make \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && \
    apt-get autoremove
RUN addgroup --gid $GID $UNAME
RUN adduser --uid $UID --gid $GID --home /home/$UNAME --shell /bin/bash --disabled-password $UNAME

WORKDIR /ct2rep

# Download the latest installer
ADD https://astral.sh/uv/install.sh /ct2rep/uv-installer.sh

# Change permissions on the installer and make it executable
RUN chown -R $UNAME /ct2rep
USER $UNAME

# Run the installer then remove it
RUN sh /ct2rep/uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/home/$UNAME/.local/bin/:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv sync

# CMD [ \
#     "python",\
#     "./src/ct2rep/CT2Rep/main.py",\
#     "--max_seq_length", "200",\
#     "--threshold", "3",\
#     "--num_workers", "20",\
#     "--batch_size", "1",\
#     "--dataset_name", "ct_dataset",\
#     "--d_model", "512",\
#     "--d_ff", "512",\
#     "--d_vf", "512",\
#     "--num_heads", "8",\
#     "--num_layers", "3",\
#     "--dropout", "0.1",\
#     "--logit_layers", "1",\
#     "--bos_idx", "0",\
#     "--eos_idx", "0",\
#     "--pad_idx", "0",\
#     "--use_bn", "0",\
#     "--drop_prob_lm", "0.5",\
#     "--rm_num_slots", "3",\
#     "--rm_num_heads", "8",\
#     "--rm_d_model", "512",\
#     "--sample_method", "beam_search",\
#     "--beam_size", "3",\
#     "--temperature", "1.0",\
#     "--sample_n", "1",\
#     "--group_size", "1",\
#     "--output_logsoftmax", "1",\
#     "--decoding_constraint", "0",\
#     "--block_trigrams", "1",\
#     "--n_gpu", "1",\
#     "--epochs", "200",\
#     "--save_dir", "results/",\
#     "--record_dir", "records/",\
#     "--save_period", "1",\
#     "--monitor_mode", "max",\
#     "--monitor_metric", "BLEU_4",\
#     "--early_stop", "50",\
#     "--optim", "Adam",\
#     "--lr_ve", "5e-5",\
#     "--lr_ed", "1e-4",\
#     "--weight_decay", "5e-5",\
#     "--amsgrad", "True",\
#     "--lr_scheduler", "StepLR",\
#     "--step_size", "50",\
#     "--gamma", "0.1",\
#     "--trainfolder", "/ct2rep/data/train",\
#     "--validfolder", "/ct2rep/data/valid"\
#     ]

