# Zero-shot Segment Anything example
This example demonstrates how to use a Python API implementation of Segment Anything pipeline inference:
- Create encoder and decoder models
- Create a zero-shot visual prompter pipeline
- Use points as prompts to learn on one image
- Segment other image using leaned on the previous image representation
- Visualized result is saved to `zsl_sam_result.jpg`

## Prerequisites
Install Model API from source. Please refer to the main [README](../../../README.md) for details.

## Run example
To run the example, please execute the following command:
```bash
python run.py <path_to_source_image> <path_to_target_image> <encoder_path> <decoder_path> <prompts>
```
where prompts are in X Y format.

To run the pipeline out-of-the box you can download the test data by running the following command from the repo root:
```bash
python tests/python/accuracy/prepare_data.py -d data
```
and then run
```bash
python run.py ../../../data/coco128/images/train2017/000000000025.jpg \
     ../../../data/coco128/images/train2017/000000000072.jpg ../../../data/otx_models/sam_vit_b_zsl_encoder.xml \
     ../../../data/otx_models/sam_vit_b_zsl_decoder.xml 464 202

```
from the sample folder. Here one prompt is passed via CLI: `(464 202)`

>*NOTE*: results of segmentation models are saved to `zsl_sam_result.jpg` file.