import random

TRAINING_DATA_FILEPATH = "text.txt"
OUTPUT_SIZE = 100


def is_end_sentence(word) -> bool:
    return word == "." or word == "?" or word == "!"


def is_punctuation(word) -> bool:
    return is_end_sentence(word) or word == ","


class TokenRepository:
    def __init__(self) -> None:
        self.id_counter: int = 0
        self.token_map = {}  # word to id map

    def register(self, word: str) -> None:
        if word in self.token_map:
            return

        self.token_map.update({word: self.id_counter})
        self.id_counter += 1

    def get_word(self, word_id: int) -> str:
        for key, val in self.token_map.items():
            if val == word_id:
                return key

        raise Exception("Id not found")

    def get_id(self, word: str) -> int:
        return self.token_map[word]


class Scanner:
    def __init__(self) -> None:
        pass

    def scan(self, repository: TokenRepository, text: str) -> list:  # list of ids of tokens
        words = text.lower().replace("\n", " ").replace(",", " , ").replace(".", " . ").replace("?", " ? ").\
            replace("-", " - ").replace("  ", " ").split(" ")
        tokens = []
        for i in words:
            repository.register(i)
            tokens.append(repository.get_id(i))

        return tokens


# all tokens seen before the current one
class SingleTokenStatistics:
    def __init__(self) -> None:
        self.predecessor_tokens = {}  # dictionary of previous tokens and their counts
        self.predecessor_count: int = 0  # total count of prev tokens

    def add(self, token: int) -> None:
        self.predecessor_count += 1
        if token not in self.predecessor_tokens:
            self.predecessor_tokens.update({token: 0})
        self.predecessor_tokens[token] += 1

    def get_weight(self, predecessor_token: int) -> float:  # returns likelihood
        if predecessor_token not in self.predecessor_tokens:
            return 0.0

        return self.predecessor_tokens[predecessor_token] / self.predecessor_count


class Predictor:
    def __init__(self) -> None:
        self.token_repository = TokenRepository()
        self.token_stats = {}  # dict of token to SingleTokenStatistics

    def train(self, text: str) -> None:
        scanner = Scanner()
        tokens = scanner.scan(self.token_repository, text)

        for i in range(0, len(tokens)):
            if i <= 0:
                continue

            curr_token = tokens[i]
            if curr_token not in self.token_stats:
                self.token_stats.update({curr_token: SingleTokenStatistics()})

            self.token_stats[curr_token].add(tokens[i - 1])

    def get_next(self, token: int) -> int:
        stats = {}
        for key, val in self.token_stats.items():
            stats.update({key: val.get_weight(token)})

        if sum(list(stats.values())) <= 0:
            return random.choice(list(stats.keys()))

        rand_items = random.choices(list(stats.keys()), weights=list(stats.values()), k=1)

        return rand_items[0]

    def tokens_to_text(self, prompt: str, tokens: list) -> str:
        text = ""
        prev = ""
        for i in tokens:
            word = self.token_repository.get_word(i)
            if is_end_sentence(prev):
                word = word.capitalize()
            if prev == "":
                prev = word
                continue
            prev = word

            if not is_punctuation(word):
                text += " "
            text += word

        return prompt + text

    def generate_text(self, text: str, length: int = 30) -> str:
        tokens = Scanner().scan(self.token_repository, text)
        generated_tokens = []
        generated_tokens.append(tokens[len(tokens) - 1])
        for i in range(length - 1):
            generated_tokens.append(self.get_next(generated_tokens[len(generated_tokens) - 1]))

        return self.tokens_to_text(text, generated_tokens)


def main():
    with open(TRAINING_DATA_FILEPATH, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    test_str = " ".join(lines)
    predictor = Predictor()
    predictor.train(test_str)
    print(predictor.generate_text(input(">"), OUTPUT_SIZE))


if __name__ == "__main__":
    main()

