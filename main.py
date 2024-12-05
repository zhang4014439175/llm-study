# This is a sample Python script.



# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def load_file():
    # Use a breakpoint in the code line below to debug your script.
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


def open_file():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])


def split_study():
    import re
    # text = "Hello, world. This, is a test."
    # result = re.split(r'(\s)', text)
    # result = re.split(r'([,.]|\s)', text)
    # print(result)
    # result = [item for item in result if item.strip()]
    # print(result)
    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)


def split_file():
    import re
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print(len(preprocessed))
    # print(preprocessed[:30])

    all_words = sorted(set(preprocessed))
    # vocab_size = len(all_words)
    # print(vocab_size)

    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        # print(item)
        if i >= 50:
            break

    from simple_tokenizer_v1 import SimpleTokenizerV1
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know," 
     Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    split_file()

