import os
from torch import nn
from config import get
from tokenizer import get_vocab_size
import torch
import embedding as E
from torch.nn import functional as F
from tokenizer.aa import START_TOKEN, END_TOKEN
class Encoder( nn.Module ):
    def __init__(self, d_model, nhead, d_ff, dropout, num_layers, activation='relu', spectrom_len=2048 ):
        super().__init__()
        self.spectrum_emb = E.SpectrumEmbedding( d_model )
        encoder_layer = nn.TransformerEncoderLayer( d_model, nhead, d_ff, dropout, activation=activation )
        self.transformer_encoder = nn.TransformerEncoder( encoder_layer, num_layers= num_layers )

    def forward(self, spectrum):
        # spectrum dim (batch, 2048, 2)
        encoded_spectrum = self.spectrum_emb( spectrum )
        out = self.transformer_encoder( encoded_spectrum )

        return out

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, num_layers, mz_positional_emb=None, activation='relu',) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dff = d_ff
        self.num_layers = num_layers

        self.precursor_emb = E.PeptidePrecursorEmbedding(d_model, mz_positional_emb)
        decoder_layer = nn.TransformerDecoderLayer( d_model, nhead, d_ff, dropout, activation=activation )
        self.decoder = nn.TransformerDecoder( decoder_layer, num_layers )

    def forward(self, ch, mz, seq, encoded_x):
        encoded_seq = self.precursor_emb( seq, ch, mz )

        out = self.decoder( encoded_seq.permute(1,0,2), encoded_x.permute(1, 0, 2) )
        return out.permute(1, 0, 2)
    


class Model( nn.Module ):

    def __init__(self) -> None:
        super().__init__()
        d_model = int(get("model.d_model"))
        n_head = int(get("model.n_head"))
        n_layers = int(get("model.n_layers"))
        d_ff = int(get("model.d_ff"))
        dropout = float(get("model.dropout"))

        self.encoder = Encoder( d_model, n_head, d_ff, dropout, n_layers )
        self.decoder = Decoder( d_model, n_head, d_ff, dropout, n_layers, self.encoder.spectrum_emb.pos_emb )
        # self.bottleneck = nn.Linear( 2048, 1 )
        self.generator = nn.Linear( d_model, get_vocab_size() )

    def forward( self, x, seq, ch, mz ):
        encoded_x = self.encoder(x)
        output = self.decoder( ch, mz, seq, encoded_x )
        # output = self.bottleneck( output.permute(0, 2, 1) ).squeeze()
        logits = self.generator( output )
        return logits

    def finish_training(self):
        self.save_model()
    
    def save_model(self):
        torch.save( self.state_dict(), 'model.pt' )
    
    def load_model(self):
        self.load_state_dict(torch.load("model.pt"))

    def load_if_file_exists(self):
        if os.path.exists("model.pt"):
            self.load_model()

def greedy_decode( model:Model, x, ch, mz, max_len = 256):
    encoded_x = model.encoder( x )

    pred = torch.ones(1, 1).fill_( START_TOKEN ).type( torch.long ).to( 'cuda' )

    for _ in range( max_len ):
    
        dec_out = model.decoder( ch, mz, pred, encoded_x ).transpose( -2, -1 )
        probs = model.generator( dec_out[...,-1] )
        _, next_token = torch.max( probs, dim= 1 )
        pred = torch.cat( [pred, next_token.reshape(-1,1)] ,dim= 1 )
        if next_token == END_TOKEN:
            break
    return pred[:,1:]

def beam_search_decode(model, x, ch, mz, max_len=256, beam_size=3):
    encoded_x = model.encoder(x)

    pred = torch.ones(1, 1).fill_(START_TOKEN).type(torch.long).to('cuda')
    sequences = [(pred, 0)]

    for _ in range(max_len):
        all_candidates = []
        
        for seq, score in sequences:
            dec_out = model.decoder(ch, mz, seq, encoded_x).transpose(-2, -1)
            probs = model.generator(dec_out[...,-1])
            topk_probs, topk_tokens = torch.topk(probs, beam_size, dim=1)

            for i in range(beam_size):
                next_token = topk_tokens[:, i].reshape(-1, 1)
                next_score = topk_probs[:, i].item()
                candidate = (torch.cat([seq, next_token], dim=1), score - next_score)
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_size]
        if all(seq[0][-1] == END_TOKEN for seq, _ in sequences):
            break

    best_seq = sorted(sequences, key=lambda tup: tup[1])[0][0]
    return best_seq[:, 1:]