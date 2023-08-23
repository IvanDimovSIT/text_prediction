import random


class TokenRepository:
    def __init__(self) -> None:
        self.id_counter = 0
        self.token_map = {}  # word to id map

    def register(self, word: str) -> None:
        if word in self.token_map:
            return

        self.token_map.update({word: self.id_counter})
        self.id_counter += 1

    def get_word(self, id: int) -> str:
        for key, val in self.token_map.items():
            if val == id:
                return key

        return None

    def get_id(self, word: str) -> int:
        return self.token_map[word]


class Tokenizer:
    def __init__(self) -> None:
        pass

    def tokenize(self, repository: TokenRepository, text: str) -> list:  # list of ids of tokens
        words = text.lower().replace("\n", " ").replace(",", " , ").replace(".", " . ").replace("?", " ? ").replace("-", " - ").replace("  ", " ").split(" ")
        tokens = []
        for i in words:
            repository.register(i)
            tokens.append(repository.get_id(i))

        return tokens


# to be used in a dict
class SingleTokenStatistics:
    def __init__(self) -> None:
        self.predecessor_tokens = {}  # dictionary of previous tokens and their counts
        self.predecessor_count = 0  # total count of prev tokens

    def add(self, token: int) -> None:
        self.predecessor_count += 1
        if token not in self.predecessor_tokens:
            self.predecessor_tokens.update({token: 0})
        self.predecessor_tokens[token] += 1

    def get_weight(self, predecessor_token: int) -> float:  # returns likelyhood
        if predecessor_token not in self.predecessor_tokens:
            return 0.0

        return self.predecessor_tokens[predecessor_token] / self.predecessor_count


class Predictor:
    def __init__(self) -> None:
        self.token_repository = TokenRepository()
        self.token_stats = {}  # dict of token to SingleTokenStatistics

    def train(self, text: str) -> None:
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(self.token_repository, text)

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

    def tokens_to_text(self, tokens: list) -> str:
        text = ""
        prev = ""
        for i in tokens:
            word = self.token_repository.get_word(i)
            if prev == "." or prev == "?" or prev == "!":
                word = word.capitalize()
            text += word
            text += " "
            
            prev = word

        return text

    def generate_text(self, text: str, length: int = 30) -> str:
        tokens = Tokenizer().tokenize(self.token_repository, text)
        generated_tokens = []
        generated_tokens.append(tokens[len(tokens)-1])
        for i in range(length-1):
            generated_tokens.append(self.get_next(generated_tokens[len(generated_tokens)-1]))

        return self.tokens_to_text(generated_tokens)


def main():
    with open("text.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    test_str = " ".join(lines)
    predictor = Predictor()
    predictor.train(test_str)
    print(predictor.generate_text(input(">"), 100))

if __name__ == "__main__":
    main()
