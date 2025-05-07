from get_data import generate_ai, get_data, split 
from src.train import train_model

def main():
    # define the config files in configs/
    # 1. download data
    # get_data.start_download()

    # 2. split the data
    # print(split.split())

    # 3. generate AI paraphrases 
    # print(generate_ai.generate_paraphrases())

    # train the model
    train_model()


if __name__ == '__main__':
    main()