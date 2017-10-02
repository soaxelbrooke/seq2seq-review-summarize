
class ModelConfig:
    """ Config object for review summary experiment """
    def __init__(self, vocab_size, use_cuda, embed_size, review_len, summary_len, batch_size,
                 context_size, start_token, attention_method):
        self.vocab_size = vocab_size
        self.use_cuda = use_cuda
        self.embed_size = embed_size
        self.review_len = review_len
        self.summary_len = summary_len
        self.batch_size = batch_size
        self.context_size = context_size
        self.start_token = start_token
        self.attention_method = attention_method
