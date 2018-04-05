# PixelDA
Try to use DA to enhance simulator data.
Current method: CycleGAN

## Usage: 
        1) build data: python build_data.py --X_input_dir "sim_images_a1_low_var" --X_output_file "sim_images_a1_low_var.tfrecords"
        2)run the network: python train_CycleGAN.py --X "tfrecordsname"

## Structure:
    PixelDA/
    ----Data/
    --------tfdata/
    ------------image0.jpeg
            	  ...
    --------features_XXX.csv
    ----random_urdfs/
    --------000/
    ------------000.urdf
        		...
        	...
    ----sim_images/
    --------sim_image000000.jpeg
        		...
    ----checkpoints/
    --------checkpoint0
        		...
    ----python files
        		


