# French formality classifier: fine-tuning CamemBERT and analysing which words and POS tags it relies on

> Individual course project · *CS372 Natural Language Processing with Python* · KAIST (exchange year), Spring 2025 · PyTorch, Hugging Face Transformers, spaCy, NLTK

The aim of this project was to classify whether a French sentence is written in formal or informal language, and then to study which words and grammatical features make a sentence more or less formal for the model. I also wanted to use the tools we had seen in the course to practise them: regular expressions to clean the data, NLTK to split it into sentences, POS tagging to analyse the results, and WordNet for an attempt at rewriting sentences in the other register.

I only had free Google Colab compute for this project, so keeping training short was an important part of the design. For this reason I fine-tuned a pre-trained French model, CamemBERT, instead of training a model from scratch. The model reaches 96.4% accuracy on the validation set. The token analysis shows that it relies on markers of spoken French such as `?`, *tu* and *ça* for informal sentences. For formal sentences, it relies on narrative verb tenses such as the *passé simple*, which is used mainly in written storytelling.

## Data

Each sentence is labelled by its source. For formal French, I used classic 19th-century novels from Project Gutenberg, which we had used in the course. For informal French, I thought spoken language was the best representative, so I used subtitles of French films, mostly comedies, and one TV episode. This choice does not perfectly separate formal from informal language, which is the main limitation discussed below.

| Class | Source | Sentences |
|---|---|---|
| Formal | 5 French classics from Project Gutenberg: *Les Misérables*, *Le Comte de Monte-Cristo*, *Madame Bovary*, *Le Rouge et le Noir*, *Les Fleurs du mal* | 32,019 |
| Informal | French subtitles of 4 films (*Intouchables*, *Les Bronzés font du ski*, *Les Tuche*, *Astérix & Obélix : L'Empire du Milieu*) and 1 TV episode (*Le Négociateur*). They are copyrighted, so they are not included (see `data/README.md`) | 8,222 |

I cleaned the subtitles with regular expressions (timestamps, HTML tags, dialogue dashes), split both corpora into sentences with NLTK, and kept 20% of the sentences for validation (seed 42).

## Model and training

I fine-tuned `camembert-base` with a two-class classification head. One reason I chose it is that its tokenizer splits words into subwords (for example, *décentralisation* becomes `▁dé`, `central`, `isation`), so what the model learns about a word part can apply to the many French words that share it. Training uses cross-entropy loss, Adam (learning rate 1e-5), batches of 16 and a maximum length of 128 tokens. To keep training short, each of the 20 epochs trains on a new random sample of 500 training sentences instead of the whole training set.

## Analysis

To measure how much a token matters for a given sentence, I replace it with `<mask>` and record how much the probability of "formal" drops. I then divide these drops by the largest one in the sentence, so the most important token gets a score of ±1 (positive means it pushes towards formal). Averaging these scores over 1,000 random sentences gives the most formal and most informal tokens. Grouping them by POS tag, using spaCy's French tagger (`fr_core_news_sm`), shows which parts of speech the model relies on.

## Results

**Accuracy.** The model reaches an accuracy of 96.4% on the full validation set (8,048 sentences, loss 0.107). Since there are four times more formal sentences, always answering "formal" would already give 79.6%, so the model removes about 80% of the remaining errors.

![20 most formal and 20 most informal tokens](figures/top_tokens.png)

<sub><i>Average masking score of each token over 1,000 random sentences. Green bars push towards formal, red bars towards informal.</i></sub>

**Tokens.** The informal tokens are mostly markers of spoken French: `?`, `!`, *Pas* (negation without *ne*), *Allez*, *Non*, *ça* and *Tu*. The formal tokens are mostly narrative verbs in the *passé simple* and imperfect, such as *murmura, voyait, entendit, alla, approcha, regardait* and *arriva*. The passé simple is almost only used in written storytelling, so these tokens indicate a 19th-century novel more than formal language. In other words, the model partly learned to separate literary narration from spoken dialogue.

**Parts of speech.** I looked at POS tags in two ways: how often each tag appears in each class, and how much the model relies on each tag.

![Share of each POS tag in formal and informal sentences](figures/pos_distribution.png)

<sub><i>Share of each POS tag among the words of 3,000 formal and 3,000 informal sentences.</i></sub>

Informal sentences contain more punctuation and pronouns, and formal sentences contain more nouns, determiners and prepositions.

![Average token-masking score per POS tag, formal vs informal sentences](figures/pos_influence.png)

<sub><i>Average masking score of the words with each POS tag, over 1,000 formal (green) and 1,000 informal (red) sentences. Positive values push towards formal.</i></sub>

Punctuation has one of the strongest effects in both classes, mostly `?` and `!` in informal sentences. In informal sentences the model also relies on adverbs, pronouns and proper nouns (often character names), and in formal sentences on verbs, nouns and adjectives. This matches the token list: narrative vocabulary on one side and markers of spoken language on the other.

**Rewriting attempt.** I also tried to rewrite sentences into the other register by replacing words with WordNet synonyms, using the classifier to choose the most formal or informal one. This did not work well, because French WordNet has few synonyms and the classifier is not reliable on single words, so this code is not included.

## Limitations

- **Formality is confounded with era and genre.** The formal sentences are 19th-century literary narration and the informal ones are film and TV dialogue from 1979 to 2024, so the accuracy partly measures "novel vs. subtitle". A fairer test would need both registers from the same period and medium, for example formal and casual emails.
- **One training run.** Training is seeded (seed 42), but it is not exactly reproducible on a GPU: two runs with the same seed on an Apple GPU gave 96.2% and 96.4%. Confident predictions stay the same, but borderline sentences can change. For example, *"Allons faire cela"* got a probability of being formal of 0.27 in the original 2025 run and 0.95 in this one.
- **Duplicate sentences.** Short lines like *"Oui!"* or *"Merci."* appear many times (about 6% of the sentences), so some identical sentences are in both the training and validation sets.
- **Approximate masking scores.** Masking one subword at a time ignores interactions between words. When the model is very confident, masking a single word barely changes its prediction, even if several words together drive the decision.
- **Tagging errors.** spaCy's small French model sometimes tags words incorrectly, and these errors carry over to the POS results.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # includes the spaCy French model
# add the 5 subtitle files to data/subtitles/ (see data/README.md), then:
jupyter notebook formality_classifier.ipynb
```

The notebook runs top to bottom in about 18 minutes on an Apple M1 Max GPU, and uses CUDA, Apple MPS or the CPU depending on what is available. With other subtitle files, the numbers will differ slightly.

## Repository structure

```
formality_classifier.ipynb   data loading, fine-tuning, evaluation, token and POS analysis (with outputs)
data/books/                  formal corpus (public domain, Gutenberg license text removed)
data/subtitles/              informal corpus: add your own .srt files (not included)
figures/                     the three figures above, taken from the notebook outputs
requirements.txt             pinned dependencies
```

## 2026 cleanup

The topic, the dataset, the model and the analysis are my own. In 2026 I cleaned and documented the notebook for publication with [Claude Code](https://claude.com/claude-code): I added relative paths and a fixed seed, fixed a word-alignment bug in the POS analysis and re-ran the notebook from scratch. The model and the training procedure are unchanged.
