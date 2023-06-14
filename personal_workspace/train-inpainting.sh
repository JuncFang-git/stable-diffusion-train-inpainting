###
 # @Author: Juncfang
 # @Date: 2023-06-12 18:07:15
 # @LastEditTime: 2023-06-13 14:37:46
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /stable-diffusion/personal_workspace/train-inpainting.sh
 #  
### 

export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/.." && pwd )"

cd $PROJECT_DIR && \
python main.py \
--train \
--name custom_inpainting_training \
--base personal_workspace/train-inpainting.yaml  \
--logdir personal_workspace/logs
