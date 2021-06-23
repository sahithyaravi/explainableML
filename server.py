from app import app

if __name__ == '__main__':
  app.run(debug=True)

# from app.config import Config
# import pandas as pd
# from sqlalchemy import create_engine
# SQLALCHEMY_DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI
# engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
# df_final_labels = pd.read_csv('yelp_dataset_cluster.csv')
# df_final_labels["round"] = 0
# df_final_labels.to_sql("yelp_dataset_cluster", con=engine, if_exists="replace")

# https://paperswithcode.com/task/sentiment-analysis