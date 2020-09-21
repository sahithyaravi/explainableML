from models.trainers import Trainer
from app.utils import clear_labels


dataset = 'yelp_dataset'   # gao, waseem or founta
t = Trainer(dataset)


#
clear_labels()
t.init_train()

# Retraining after user updates labels
#t.retrain()

