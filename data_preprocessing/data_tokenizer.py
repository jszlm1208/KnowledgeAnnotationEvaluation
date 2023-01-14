from tokenizers import Tokenizer

output = tokenizer.encode("A question: don't you think that Fatima was subjected to sorcery by this al - Taymani?")
print(output.tokens)
