from get_data import generate_ai, get_data, split 

def main():
    # 1. download data
    # get_data.start_download()

    # 2. split the data
    # Created → test: 400, train: 1360, val: 240
    # print(split.split())

    # 3. generate AI paraphrases 
    print(generate_ai.generate_paraphrases())

    # train the model

    # test the model on test dataset

    # log the prints on separate report file


if __name__ == '__main__':
    main()