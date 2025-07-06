# Pair-GRPO Training
First please clone our repo and prepare the python environment for inference. We recommend using Python>=3.10. 
```
git clone https://github.com/wendell0218/FocusDiff.git
cd FocusDiff

conda create -n focusdiff python=3.11
conda activate focusdiff
pip install -r requirements.txt
```

Then, below is the organized file structure with brief descriptions:

```text
openr1-focusdiff/
├── recipes/                        # Configuration files related to GRPO-based training
├── src/
    └── open_r1                     # Source code for the Pair-GRPO
        ├── internvl/               # Inference code related to InternVL
        ├── models/                 # Model architecture of Janus-FocusDiff
        ├── configs.py              # Configuration settings for Pair-GRPO
        ├── grpo_trainer_pair.py    # Implementation of Pair-GRPO for text-to-image generation
        ├── grpo_pair.py            # Entry point for running Pair-GRPO
        ├── grpo_trainer_vanilla.py # Implementation of vanilla GRPO for text-to-image generation
        ├── grpo_vanilla.py         # Entry point for running vanilla GRPO
        ├── internvl_img.py         # Script for reward calculation with InternVL
        ├── mydataset.py            # Dataset construction
        └── llama.py                # Inference script for Janus-FocusDiff
```

Next, it is necessary to download some essential backbone models as the policy/reference model, as well as VLMs that serve as the reward models.
Specifically, you can download [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) as the backbone model, filling in the local path of the Janus-Pro-7B model weights in the necessary code ([recipes/t2i/grpo.yml#L2](recipes/t2i/grpo.yml#L2), [src/open_r1/grpo_trainer_pair.py#L207](src/open_r1/grpo_trainer_pair.py#L207), [src/open_r1/grpo_trainer_pair.py#371](src/open_r1/grpo_trainer_pair.py#371)).

While for rewars models, there are two optinal ways to call them: online and offline.
For models that are deployed on GPUs with larger memory, such as [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B), online deployment can be adopted, and a URL can be exposed as the calling interface.
For models that are deployed on GPUs with relatively smaller memory, such as [InternVL2.5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B), offline deployment can be used, where the model is directly deployed on the GPUs for training.
This eliminates the need for network communication and provides greater training stability.
Specifically, for the former, after model deployment is complete, the URL path needs to be modified in the corresponding configuration, e.g., [src/open_r1/grpo_pair.py#L101](src/open_r1/grpo_pair.py#L101). 
For the latter, the local weight path of the reward model should be filled in [src/open_r1/internvl_img.py#L98](src/open_r1/internvl_img.py#L98).
Moreover, the hyperparameter ``internvl_tp`` in the [configuration file](recipes/t2i/grpo.yml) is used to control the parameter weights of the reward model. A value of ``26b`` corresponds to the reward model of InternVL2.5-26B, indicating an offline deployment; a value of ``8b`` corresponds to the reward model of InternVL2.5-8B, indicating an online deployment.

Finally, after preparing some training data-related prompt files (optionally including some groundtruth images), you can proceed with the GRPO (Pair-GRPO) training for text-to-image generation under the AR paradigm using the following script:

```bash
cd FocusDiff/openr1-focusdiff/src/open_r1

export ACCELERATE_CONFIG=../../recipes/accelerate_configs/zero2.yaml
export GRPO_CONFIG=../../recipes/t2i/grpo.yaml
export NUM_PROCESSES=8

# training command for Pair GRPO
accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes $NUM_PROCESSES \
  grpo_pair.py \
  --config $GRPO_CONFIG
# training command for vanilla GRPO
accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes $NUM_PROCESSES \
  grpo_vanilla.py \
  --config $GRPO_CONFIG
```
