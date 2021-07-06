from tokenizers.models import WordPiece
from tokenizers import Tokenizer, decoders
from tokenizers.trainers import WordPieceTrainer

tokenizer = Tokenizer(WordPiece())
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(special_tokens=[ "[UNK]", "[PAD]","[CLS]", "[SEP]"],
                           vocab_size=8000, min_frequency=0, continuing_subword_prefix="##")
tokenizer.decoder = decoders.WordPiece()

tokenizer.train(files=['clean_wikiv2.txt'], trainer=trainer)

from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

tokenizer.save("tokenizer.json")
