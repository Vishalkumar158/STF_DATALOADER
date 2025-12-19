This code is for loading [Seeing Through Fog (STF)](https://github.com/princeton-computational-imaging/SeeingThroughFog) dataset.

# Setupping environment to run this code 
- if your system has CUDA Comilation tools is 12.6 then just run 

```
    conda env create -f environment.yml
```

- If your System dont have CUDA Comilation tools is 12.6 then follow these steps:
  ```
        # 1. Create a new conda environment (Python 3.10 recommended for best wheel support)
        conda create -n openpcdet_cuda12 python=3.10 -y
        conda activate openpcdet_cuda12

        # 2. Install PyTorch with CUDA 12.4 support (latest stable as of Dec 2025)
        conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

        # Or via pip (same thing)
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

        # 3. Install other requirements from setup.py (updated versions)
        pip install numba llvmlite tensorboardX easydict pyyaml scikit-image tqdm

        # 4. Install spconv prebuilt for CUDA 12.6
        pip install spconv-cu126

        # If not available, try:
        pip install spconv-cu124   # also works due to minor version compatibility

        # Or latest generic (it will pick closest):
        pip install spconv --upgrade

        # 5. Fix SharedArray (common source of numpy errors)
        # Old SharedArray versions conflict with modern numpy
        pip uninstall SharedArray -y
        pip install SharedArray==3.2.1   # or try 3.2.0 if issues

        # Alternative if still errors: some forks remove SharedArray dependency
        # (only needed for Waymo multi-process loading; safe to skip for KITTI/STF)

        # 6. Compile OpenPCDet custom ops
        python setup.py develop

        # Note: You may need build tools
        conda install -c conda-forge ninja cmake -y   # helps compilation

        # Installing quaternion
        pip install quaternion

    ```

# Tp split data in train and val text file run : 
    
         python split.py
    

# To generate pkl file for STF Dataset just run : 
    
          python STF_Dataloader.py create_stf_infos /home/saksham/samsad/mtech-project/datasets/SeeingThroghFog/pcdet/stf_config.yaml
    

- Note this code is inspired from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and a  [forkedrepo] (https://github.com/barzanisar/OpenPCDet)
