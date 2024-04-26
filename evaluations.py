from evaluate import load


def main():

    mauve = load('mauve')

    predictions = ["hello world", "goodnight moon"]
    references = ["hello world",  "goodnight moon"]

    # TODO: Load webtext dataset for reference
    mauve_results = mauve.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    main()
