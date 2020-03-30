from models.trainers import Trainer
from app.utils import clear_labels


dataset = 'davidson_dataset'   # gao, waseem or founta
t = Trainer(dataset)


# init
# clear_labels()
# t.init_train()

# Retraining after user updates labels
#t.retrain()

