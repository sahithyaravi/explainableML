from models.trainers import Trainer

# Initial training
dataset = 'davidson_dataset'   # gao, waseem or founta
t = Trainer(dataset)
t.init_train()

# Retraining after user updates labels
t.retrain()

