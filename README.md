# Multi-Context Fusion Transformer for Pedestrian Crossing Intention Prediction in Urban Environments
![架构](https://github.com/user-attachments/assets/27011086-a1d2-4bef-b599-13f95463c244)
# Evaluate the model using our trained weights
PIE dataset
python test.py Weight_PIE/

JAAD (all subset)
python test.py Weight_JAAD_all/

JAAD (behavior subset)
python test.py Weight_JAAD_beh/

#  Train the model from scratch, for example
python train_test.py -c config_files/nonvisual/NonVisualModel.yaml
