# -*- coding: utf-8 -*-
import os
from json import dump, load
from time import time

from token2token.utils import build_dataset, get_savedir
from token2token.tokenization import load_word_tokenizer, get_vocab, update_dicts
from token2token.methods import rerank, rerank_mp, get_trans_pmi


class Word2word:
    """The word2word class.

    Usage:
        from word2word import Word2word

        # Download and load a pre-computed bilingual lexicon
        en2fr = Word2word("en", "fr")
        print(en2fr("apple"))
        # out: ['pomme', 'pommes', 'pommier', 'tartes', 'fleurs']

        # Build a custom bilingual lexicon
        # (requires two aligned files, e.g., my_corpus.en, my_corpus.fr)
        my_en2fr = Word2word.make("en", "fr", "my_corpus")
    """

    def __init__(self, lang1=None, lang2=None, path=None):
        """Loads this object with a custom-built bilingual lexicon.

        savedir is the directory containing {lang1}-{lang2}.pkl files
        built from the make function.
        """
        if not path:
            if lang1 and lang2:
                savedir = get_savedir()
                path = os.path.join(savedir, f"{lang1}-{lang2}.json")

            else:
                 raise ValueError("you have to define either correct path or lang1 and lang2.")

        assert os.path.exists(path), f"processed lexicon file not found at {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = load(f)

        self.lang1 = data["src_lang"]
        self.lang2 = data["tgt_lang"]

        print(f"Loaded word2word custom bilingual lexicon from {path}")

        self.word2x = data["src_vocab"]
        self.word2y = data["tgt_vocab"]
        self.y2word = {y:x for x,y in self.word2y.items()}

        # Rebuild translations into list of (target, score) tuples
        x2ys = {}
        for src, entries in data["translations"].items():
            l = []
            for entry in entries:
                key = next(iter(entry))
                l.append((self.word2y[key], entry[key]))

            x2ys[self.word2x[src]] = l
        self.x2ys = x2ys

    def __call__(self, query, n_best=5):
        """Retrieve top-k word translations for the query word."""
        try:
            x = self.word2x[query]
            ys = self.x2ys[x]
            words = {self.y2word[y[0]] : y[1] for y in ys[:n_best]}
        except KeyError:
            raise KeyError(
                f"query word {query} not found in the bilingual lexicon."
            )
        return words

    def __len__(self):
        """Return the number of source words for which translation exists."""
        return len(self.x2ys)

    def compute_summary(self):
        """Compute basic summaries for the bilingual lexicon."""
        n_unique_ys = len(set([y for ys in self.x2ys.values() for y in ys]))
        n_ys = [len(ys) for ys in self.x2ys.values()]
        self.summary = {
            "n_valid_words": len(self),
            "n_valid_targets": n_unique_ys,
            "n_total_words": len(self.word2x),
            "n_total_targets": len(self.y2word),
            "n_translations_per_word": sum(n_ys) / len(n_ys),
            "n_sentences": None,  # original file required
        }
        return self.summary

    @classmethod
    def make(
            cls,
            lang1: str,
            lang2: str,
            datapref: str = None,
            n_lines: int = 1000000,
            cutoff: int = 5000,
            rerank_width: int = 100,
            rerank_impl: str = "multiprocessing",
            n_translations: int = 10,
            save_pmi: bool = False,
            savedir: str = None,
            num_workers: int = 16,
    ):
        """Build a bilingual lexicon using a parallel corpus."""

        print("Step 1. Load tokenizers and build dataset")
        lang1, lang2 = sorted([lang1, lang2])
        tokenizer1 = load_word_tokenizer(lang1)
        tokenizer2 = load_word_tokenizer(lang2)
        dataset = build_dataset(lang1, lang2, tokenizer1, tokenizer2)

        # input savedir if provided, else datapref (custom data location);
        # system default otherwise
        savedir = get_savedir(savedir if savedir else datapref)

        print("Step 3. Compute vocabularies")
        # word <-> index

        word2x, x2word, x2cnt = get_vocab(dataset.take(n_lines), lang1)
        word2y, y2word, y2cnt = get_vocab(dataset.take(n_lines), lang2)

        print("Step 4. Update count dictionaries")
        # monolingual and cross-lingual dictionaries
        x2xs, y2ys, x2ys, y2xs, seqlens1, seqlens2 = update_dicts(
            dataset.take(n_lines), lang1, lang2, word2x, word2y, cutoff, n_lines, save_pmi
        )

        t0 = time()
        print("Step 5. Translation using CPE scores")
        if rerank_impl == "simple":
            x2ys_cpe = rerank(x2ys, x2cnt, x2xs, rerank_width, n_translations)
            y2xs_cpe = rerank(y2xs, y2cnt, y2ys, rerank_width, n_translations)
        elif rerank_impl == "multiprocessing":
            x2ys_cpe = rerank_mp(
                x2ys, x2cnt, x2xs, rerank_width, n_translations, num_workers
            )
            y2xs_cpe = rerank_mp(
                y2xs, y2cnt, y2ys, rerank_width, n_translations, num_workers
            )
        else:
            raise ValueError("unrecognized --rerank_impl argument. "
                             "Options: simple, multiprocessing")
        print(f"Time taken for step 5: {time() - t0:.2f}s")

        print("Saving...")
        Word2word.save(lang1, lang2, savedir, word2x, word2y, x2word,
                       x2ys_cpe, y2word, y2xs_cpe)

        if save_pmi:
            print("Step 5-1. Translation using PMI scores")
            subdir = os.path.join(savedir, "pmi")
            os.makedirs(subdir, exist_ok=True)
            Nx = sum(seqlens1)
            Ny = sum(seqlens2)
            Nxy = sum([seqlen_x * seqlen_y
                       for seqlen_x, seqlen_y in zip(seqlens1, seqlens2)])

            x2ys_pmi = get_trans_pmi(x2ys, x2cnt, y2cnt, Nxy, Nx, Ny,
                                     rerank_width, n_translations)
            y2xs_pmi = get_trans_pmi(y2xs, y2cnt, x2cnt, Nxy, Ny, Nx,
                                     rerank_width, n_translations)

            Word2word.save(lang1, lang2, subdir, word2x, word2y, x2word,
                           x2ys_pmi, y2word, y2xs_pmi)

        print("Done!")
        return cls(lang1, lang2, word2x, y2word, x2ys_cpe)

    @staticmethod
    def save(lang1, lang2, savedir, word2x, word2y, x2word, x2ys, y2word, y2xs):

        def _dump_json(path, src_vocab, tgt_vocab, translations, src_lang, tgt_lang,
                    id2word_src, id2word_tgt):
            """Helper to write bilingual dictionary JSON with words instead of IDs."""
            norm_translations = {}
            for src_id, tgts in translations.items():
                if not tgts:
                    norm_translations[id2word_src[int(src_id)]] = []
                    continue
                
                norm_translations[id2word_src[int(src_id)]] = [
                    {id2word_tgt[int(tgt)]: float(score)}
                    for tgt, score in tgts
                ]

            data = {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "translations": norm_translations
            }
            with open(path, "w", encoding="utf-8") as f:
                dump(data, f, ensure_ascii=False, indent=2)

        # lang1 → lang2
        _dump_json(
            os.path.join(savedir, f"{lang1}-{lang2}.json"),
            src_vocab=word2x,
            tgt_vocab=word2y,
            translations=x2ys,
            src_lang=lang1,
            tgt_lang=lang2,
            id2word_src=x2word,
            id2word_tgt=y2word
        )

        # lang2 → lang1
        _dump_json(
            os.path.join(savedir, f"{lang2}-{lang1}.json"),
            src_vocab=word2y,
            tgt_vocab=word2x,
            translations=y2xs,
            src_lang=lang2,
            tgt_lang=lang1,
            id2word_src=y2word,
            id2word_tgt=x2word
        )
