# Changelog

## 0.14.0 — 2026-11-27

The new model calls below need Kaimei API 1.5.0; against an older API they raise `NotFoundError`.
`explanations.compare_methods` uses only calls that already existed, and everything else works as it did.

### Added

- `client.models.reference_data(model, sample=...)` gives each sampled row's stored label beside it, as
  `sample_labels` (API 1.5.0 sends it): one entry per row of `sample`, `None` where a row was stored
  without a label. Nothing changed in the call; the method now says what it returns.
- `client.models.monitoring(model, window="day", days=30, feature=None, trailing=500, points=1500)`: a model's
  logged traffic over time, with the drift a check would have found at the end of every hour, day or week, the
  PSI chance alone gives each input in a window of that size (`levels`), and one input drawn in full against the
  reference data.
  `kaimei.display.rows(history)` and `history.to_frame()` give one row per window, with each input's PSI as a
  `drift.<input>` column.
- `client.models.population(model, days=30, limit=2000)`: a model's logged predictions one by one, with the
  fields they can be grouped by (inputs, metadata keys, what was predicted, what happened) ranked by how far each
  separates the model's answers. `kaimei.display.rows(population)` and `population.to_frame()` give one row per
  prediction with a column per field, ready for `groupby`.
- `client.models.similar(model, input=None, prediction_id=None, k=10, source="reference")`: the rows Kaimei
  already holds that are most like one case (its inputs, or a logged prediction), from the model's reference rows
  or its newest logged predictions (`source="predictions"`), nearest first, with what is known about each (a
  reference row's label; a logged prediction's answer and recorded outcome), the inputs each differs on most, and
  a `summary` that counts them and says it in a sentence (`POST /v1/models/{model}/similar`, in the same API
  release). Nothing is run and nothing is stored. `kaimei.display.rows(similar)` and `similar.to_frame()` give one
  row per case with an `input:<name>` column per input.
- `client.models.holdings(model)`: what Kaimei holds for a model and what stays with you, read from what is
  stored (`GET /v1/models/{model}/holdings`, in the same API release): who runs the model, each thing Kaimei
  could hold (weights and structure, schema, reference rows, logged predictions, outcomes, explanations, insights,
  a credential for your endpoint) with `held`, a `count`, `bytes` and a `detail`, and what crosses between your
  network and Kaimei when a prediction, an explanation or an analysis is asked for.
  `kaimei.display.rows(holdings)` and `holdings.to_frame()` give one row per item.
- `client.models.case_study(model)`: a model written up as a case study from what Kaimei already holds about
  it (`GET /v1/models/{model}/case-study`, in the same API release): a `headline`, the key `facts`, ten `sections`
  each with a `summary` built from the numbers, a `status` (`ready`, `stale` or `missing`) and the data a chart
  needs, and `gaps`, what is not on file and what would add it. Nothing is run and nothing counts against a
  limit. `client.models.case_study_markdown(model)` returns the same as a Markdown document with no HTML.
  In a notebook a case study shows as a card, and `kaimei.display.rows(study)` and `study.to_frame()` give one
  row per section.
- `client.models.compare(model, other, source="reference", limit=500)`: two models that take the same inputs, run
  on the same rows (`GET /v1/models/{model}/comparison`, in the same API release): how often they decide alike,
  the correlation and gap between their answers, each one's accuracy and AUC (or MAE and R²) where the truth is
  known, the rows they differ on most and, when both have it stored, their global importance side by side.
  `kaimei.display.rows(comparison)` and `comparison.to_frame()` give one row per row compared, with each model's
  value and decision, the `gap` and an `input:<name>` column per input. It is a read, with an hourly allowance of its own.
- `client.explanations.compare_methods(model, input, methods=None)`: one input explained once with each method
  the model lists (`explain_methods`, without `auto`), and what the methods say together. It loops
  `explanations.create`, so each method is one explanation of the month's allowance; no prediction is logged
  (`prediction_id=` attaches them all to one already in the log, `options=` goes to every call). The answer has
  the `explanations`, an `agreement` (a `sentence` that follows the numbers, such as "All three put credit_score
  first and agree on the top three.", Spearman's `rank_correlation` averaged over the pairs of methods, the input
  each puts `first`, the inputs they push `opposed` ways), `rows` (one per input any method ranks in its top
  `top=5`, with each method's attribution and rank) and `costs` (each method's `n_evals`, `latency_ms` and whether
  it is `exact`). It works with any API that has the methods; nothing new is asked of the server.
- The records these calls return show as Kaimei cards when they end a notebook cell, as an explanation does. Each
  leads with the sentence a person would say about it, then the few numbers behind it, one small picture and a
  link to the model's page in the dashboard. Like the other cards they are plain elements styled inline (so they
  keep their layout on GitHub), escape every name the API or you gave, follow `kaimei.set_display("raw")`, and
  render what they can of an older or partial record.
  - A monitoring history: the latest window's reading in words (stable, within chance, moderate or significant
    drift) with the leading input's PSI beside its own chance level, one bar per window toned by its reading with
    a line at each threshold, and the inputs ranked by peak PSI. A model with no drift reading draws its traffic.
  - A population: how many predictions, the field that separates the answers most, and that field's groups with
    their rows and the classes predicted in each as a split bar (the mean answer as a bar for a regression).
  - A comparison of two models: how many rows they decide the same way (or how far their answers correlate), the
    mean and largest gap, each model's score with the rows it is measured on, and the three rows they differ on
    most with both answers drawn on one line.
  - Similar cases: the summary sentence, then each case with a similarity bar, what is known of it (its label, or
    what was predicted and the recorded outcome) and the inputs it differs on most beside the asked values.
  - What Kaimei holds: who runs the model, then "Stays with you" beside "Held by Kaimei" with counts and sizes.
  - A comparison of explanation methods: whether they agree, their rank agreement, and each method's top three
    inputs as bars on one shared scale with its model calls, its time and whether it is exact.

### Changed

- A model report's card leads with its headline in the same type as the other cards' opening sentence.
- A record whose card cannot be drawn shows its plain representation in the cell (`<Kaimei monitoring>`) instead
  of raising there. `kaimei.display.html(record)` still raises.

- A drift insight whose score the reference data's own sampling can reach (the API's `beyond_chance: false`, after
  1.3.1) is shown as `within chance` in a notebook, and its bars are drawn without a verdict's colour. Insights
  from an older API show as before.

## 0.13.0 — 2026-11-26

Everything below needs Kaimei API 1.4.0. Against an older API these calls raise `NotFoundError`; everything else
works as it did.

### Added

- **`client.decisions`**: decision records, for the people who answer for a model's decisions.
  `retrieve(decision_id)` returns one logged decision and how the model reached it, found by the id it was logged
  under: the outcome, each input's effect with the sentence that states it, what would have changed the outcome,
  how it was computed, what a record cannot show, and a SHA-256 `fingerprint` of the stored facts.
  `explain(decision_id)` explains a decision that is already logged, as it was logged, and returns its record.
  `list(model=, q=, explained=)` and `iterate()` find decisions; `pdf(decision_id)` returns the record as a PDF
  and `save_pdf(decision_id, path)` writes it. An id keeps its slashes in the address; everything else is escaped.
- **A decision record shows as a card in a notebook**, with the API's own sentences, and `to_frame()` gives one
  row per input's effect. A page of decisions shows as a table. A text model's record shows the text token by token,
  each token on the colour of its effect; a picture's record says where the model looked, and its PDF
  (`decisions.save_pdf`) holds the picture with the evidence drawn over it.
- **`client.models.evidence(model)`** returns what Kaimei holds for a model against the requirements its guides
  cover, requirement by requirement, with what would add the evidence that is missing. **`client.models.report`**
  returns the model documentation report as data, `report_pdf` as a PDF and `save_report_pdf` writes it. Both show
  as cards in a notebook, and `to_frame()` gives one row per requirement, or per finding.
- **`client.decisions.record_policy(decision_id, outcome=, rules=)`** records what was finally decided and the
  rules applied outside the model (a cut-off, an overlay rule, a person's review); `remove_policy` removes it. A
  decision record gains `policy` (the outcome and each rule in a sentence, as recorded), `holds` and `history`,
  and its card shows them. `record.explanation.reference.digest` is a SHA-256 of the reference rows an explanation
  of a table was drawn from.
- **`client.retention`**: `retrieve()` returns the retention schedule model by model, with what is due and what
  is held; `set(keep_days)`, `set_for_model(model, keep_days)` and `clear_for_model(model)` change it;
  `run(dry_run=)` applies it now or says what it would delete. `place_hold(decision= or model=, reason=, matter=)`
  keeps records whatever the schedule says, `holds()` lists them and `release_hold(id)` releases one.
- **`client.decisions.delete(decision_id, reason=)`** deletes a logged decision with its explanations, and
  **`client.explanations.delete(id)`** one explanation. Both are refused while a hold covers the record.
- **`client.audit_log`**: `list(action=, actor=, target_type=, target_id=, model=, since=, until=)` and
  `iterate()` read who did what in the organization; `csv()` and `save_csv(path)` export it. A page of entries
  shows as a table, and `to_frame()` gives one row per entry.
- **`client.reason_tables`**: the approved wording for why a model decided against someone. `save_draft(model,
  adverse_outcome=, entries=, never_state=)` creates or replaces a model's draft, `approve(table_id)` puts it in
  force (a signed-in reviewer, admin or owner), `in_force(model)` and `draft(model)` read them, `list` gives every
  version, and `discard` deletes a draft. A decision record's `reasons` holds the draft reasons drawn with the
  version in force when the decision was made; `decisions.retrieve(id, reason_table="draft")` previews a draft's
  wording on a real decision. Both show as cards in a notebook.
- **The `reviewer` role** can be given in `organization.invite` and `organization.update_member`: it reads
  everything, can call `decisions.explain`, and drafts and approves reason tables; nothing else that writes.

## 0.12.0 — 2026-11-25

What is new needs Kaimei API 1.3.0, which adds model graphs (structures of `family: "graph"`), tensors of whole
numbers, tokenizers, generation and agents: deploy the API first. The calls 0.11.0 had work as they did, against
API 1.1.0 or 1.2.0, with these differences:

- With `onnx` installed (and `skl2onnx`, for scikit-learn), `via="auto"` reads a model Kaimei's own readers refuse
  or only draw, and a PyTorch module they cannot call, as a graph, which those APIs refuse. Nothing is made on
  the API: `models.upload` then uploads the model as Kaimei's own readers read it, as 0.11.0 did (refused, or
  drawn and not run), with a warning that API 1.3.0 runs it. `via="onnx"`, an ONNX model given directly,
  `models.upload_structure`, and an upload described with `inputs`, `output`, `tokenizer` or `mask_layout` (which
  only a graph is read with) raise an error that names API 1.3.0 instead. So does an upload whose task, feature
  types or agent those APIs do not know, where the API's own validation error would have been raised.
- `upload_tensors` sends an array of whole numbers or flags in its own type (`i64`, `bool` and the like), which
  those APIs refuse with an error that names API 1.3.0. `dtype="f32"` sends it as 0.11.0 did.
- `on_unsupported="error"` refuses every reading Kaimei would only draw, on every route: also a network whose
  reading does not give the model's own outputs (a module that branches on its input, say), a classifier whose
  class Kaimei would report differently from its own `predict()`, and a summary of feature importances, which
  0.11.0 returned. `weights="summary"` is not refused for it. `models.upload(on_unsupported="error")` also refuses
  a model the API says it only draws: it deletes the model again before raising, and says so.
- Reading a PyTorch module with Kaimei's own readers (`via="auto"` or `"native"`) leaves each of its parts in its
  own training mode. 0.11.0 left the module in eval mode.
- With `weights="summary"`, the upload's warning says that no weights were sent and that an agent can run the
  model; it said the model was stored as a sketch or feature-importance summary.

### Added

- **Kaimei works out what a model is.** The task from `labels=`, a single sigmoid, or a Hugging Face model's class and
  config (classifiers with their `id2label`, multi-label ones, language models, encoder-decoders, segmenters,
  detectors, pooled encoders as embeddings); every input a feature of its own (texts with their masks by name, ids as
  integer columns ranged by their embedding tables, single numbers as columns, long runs and recordings as tensors or
  `audio`); a Hugging Face language model's cache turned off for its export; a diffusers pipeline's scheduler as its
  `sampler`. An example input the features cannot give back is refused (`param="inputs"`), and so are labels of the
  wrong count on the ONNX route. A scikit-learn classifier fitted on a table of yes-or-no answers (a
  `OneVsRestClassifier`, an `MLPClassifier`, a `RidgeClassifier`, alone or at the end of a pipeline) is a
  `multilabel_classification`, each label's probability as the model gives it (a ridge classifier's scores each
  through a sigmoid); a classifier with several outputs of its own (a `MultiOutputClassifier`, a forest or a k-NN
  whose `classes_` is a list) is refused with that reason. A `task=` that contradicts the model
  (`"multiclass_classification"` for a multi-label classifier) is refused on every route, saying what to pass.
- **More frameworks.** TensorFlow SavedModels and `.tflite` files (tf2onnx), JAX and Flax NNX (jax2onnx), `.gguf` files
  (opened by transformers), and Keras 3 models with recurrent layers (through `tf.function`). A JAX model is
  converted for one row, whatever the number of rows of `sample`, and a Keras model with several outputs is checked
  on the output `output=` picks.
- **Pipelines read exactly.** scikit-learn bag-of-words text pipelines, and pipelines that start with a
  `ColumnTransformer` (or an encoder straight into the model), are read with their preprocessing as part of the
  model, so the features are the text or the table's own columns.
  - **Bag of words.** `TfidfVectorizer`, or `CountVectorizer` and `TfidfTransformer`, then a model: a linear model
    or a naive Bayes is scored with its own coefficients, and any other model after the counts (a forest, an SVM,
    a linear `SVC`, a model made sparse with `sparsify()`), with steps between the counts and it (`TruncatedSVD`,
    `SelectKBest`, a scaler), is converted by `skl2onnx`, joined on and checked. `use_idf=False`, sub-linear
    scaling, norms, bigrams with stop words and a vocabulary of your own are followed. Texts (`X=`) may be a list,
    a pandas Series, a NumPy array or a table of one column, and are in the check as you wrote them.
  - **A bag of words is never read for the start of a text.** It counts every word, so the pipeline is read for
    4,096 tokens (`text_pipelines.DEFAULT_TEXT_TOKENS`), or twice the longest text given up to the API's 32,768, or
    `max_length=`, and its feature says `"longer": "refuse"`: Kaimei API 1.3.0 refuses a longer text with an error
    instead of cutting it. The summary says so.
  - **What is refused, with the reason:** a vectorizer with `strip_accents="ascii"` (it also deletes every
    character that is not ASCII, which Kaimei's word tokenizer keeps; refit with `"unicode"`), a character
    analyzer, a token pattern other than scikit-learn's default, a `HashingVectorizer` (it keeps no vocabulary), a
    vocabulary word longer than 256 characters, a pipeline whose last step is not a model, and any pipeline whose
    words Kaimei's tokenizer would find differently from the vectorizer on your texts and Kaimei's sample texts.
  - **Tables.** A `ColumnTransformer` of one-hot or ordinal encoders, scalers and `SimpleImputer`s, each kind of
    column in a `Pipeline` of its own or not, in front of any model the ONNX route reads. A column an encoder reads
    becomes a `categorical` feature with the encoder's categories (numbers and flags named by their text, `"3"`,
    `"True"`); infrequent categories (`min_frequency`, `max_categories`) and `drop` come out as scikit-learn groups
    and drops them. Columns may be chosen by name, position, a boolean mask or a slice. A missing value is sent as
    `null`: the model is given what the imputer fills in, the feature's `baseline`. Transformer weights,
    `MinMaxScaler(clip=True)`, scalers after an encoder and a transformer as the remainder are followed. A part
    Kaimei does not write down (`KBinsDiscretizer`, `PolynomialFeatures`, a `FunctionTransformer` of your own) is
    refused by name when columns are chosen by name, and converted whole by `skl2onnx` when they are chosen by
    position, as is a column read by two parts or `SimpleImputer(add_indicator=True)`.
  - **Numbers far from zero.** The columns are computed in double precision, as scikit-learn computes them, the
    pipeline is asked about your rows as the doubles they are, and made-up rows of the check are drawn from the
    range of the fitted data. The model after the columns is read in single precision only when that gives the
    pipeline's numbers on those rows, else in double (`metadata.model_precision`).
  - `feature_names=` names the columns of a pipeline fitted on an array (one name for each column of the array). Its
    structure says which column each feature is read from (`metadata.table_columns`), and `models.upload` stores
    reference rows from an array under the right features.
- **Tasks:** `multilabel_classification`, `object_detection` (`output={"boxes", "scores", ...}`), `segmentation`,
  `clustering` and `sequence_to_sequence`. `upload(sampler=...)`; `generate(seeds=..., draw=...)`.
- **A segmentation model's class axis is read off its graph.** The scores are followed back from the output to the
  step that makes the classes (a convolution's channels, a linear layer's last axis) through every step that keeps
  the axis where it is or moves it knowably: anything worked out value by value (a log-softmax, a skip sum, a
  scale, a clip), a resize, a pad, a crop, a transpose, a squeeze, an unsqueeze, and a reshape that leaves the axis
  whole. An argmax anywhere on the way makes the output class numbers, whatever follows it (a cast). When no step
  says, the labels do (the one axis as long as `labels`), and then the shape (of the three axes after the row axis,
  the one whose size differs from two equal others). Only an output none of these settles is refused, saying to
  pass `mask_layout="chw"` (class scores, then height and width), `"hwc"` (height, width, then class scores) or
  `"indices"` (a class number for each pixel), a new argument of `models.upload`, `models.upload_structure` and
  `extract_structure`. An output whose classes the graph puts where no mask keeps them (between the height and the
  width, or folded into another axis by a reshape that is not undone) is refused with that reason. A model with
  several outputs is read without `output=`: of the outputs that can hold a mask, the one whose height and width
  are the input picture's comes first, then one at a whole fraction or multiple of it, then the rest; among equals,
  floats come before whole numbers, then the file's order (torchvision's FCN and DeepLab give `out` and `aux`; the
  scores of each class for each of a hundred queries are not taken for the mask because they come first; a
  fraction of the picture is the same fraction down and across, so a hundred boxes are not taken for one either).
  A conversion from Keras, a TensorFlow file or JAX, which compares the library's own answer, reads the first
  output that can hold a mask. One class picked out of several with its axis dropped (a slice or a split to one
  class and a squeeze, or a `Gather` of one place, `x[:, 1]`) is one score for each pixel, read `chw`: its
  probability is cut at one half, and its raw score, which says nothing of where that class beats the others, is
  refused naming `threshold` and read where `threshold=` says. A height and a width of floats that no step explains is
  never taken for class numbers because one row's values are whole: every check row is asked, and floats that are
  whole on all of them are refused, naming `mask_layout`.
- **A segmentation of one score for each pixel is cut where the model's own probability is one half.** A model
  that ends in a sigmoid gives probabilities, cut at 0.5; one that gives raw scores (no sigmoid in the file, as
  most U-Nets are exported) is cut at 0, which is where a sigmoid would make them one half. Kaimei reads which it
  is off the graph: back from the output through the steps that only move scores (a resize, a transpose, a
  squeeze, a slice, a pooling), a sigmoid or a softmax means probabilities, and a convolution, a linear layer, a
  normalisation or a sum of those means raw scores; after any other step the model's own values on its check rows
  say. The structure's head carries the cut (`head.threshold`) and what the scores were taken for
  (`head.score_kind`), and the model's summary on Kaimei says it. The same scores give the same mask in every form
  a file writes them (`1 x 1 x H x W`, `1 x H x W`, `H x W x 1` and `H x W`), a score exactly at the cut being the
  foreground; a height and a width of whole numbers is a class number for each pixel. The scores of a model that ends
  in `squeeze(0)` as PyTorch writes it when the rows are left open are followed through that branch, so `C x H x W`
  and `H x W x C` outputs of pictures that are not square are read. Scores that are never negative and go above 1
  are neither raw scores nor probabilities, and the model is refused, naming `threshold`, with the range seen; a
  final product or quotient with a constant above nought moves the cut with it (a sigmoid times 255 is cut at
  127.5, with no `score_kind`). A constant added before a clip or a scale (a hard sigmoid written out,
  `relu6(x + 3) / 6`) leaves the cut to the model's own values. One column picked from a two-class classifier of
  an ONNX file is read as a margin only where the classifier gives margins (a linear model or a support vector
  machine without probabilities of its own); a column that is already a probability is read as the number it is.
- `threshold=` on `models.upload`, `models.upload_structure`, `extract_structure`, `graphs.from_onnx` and
  `graphs.via_onnx` (and `kaimei.onnx_graph.import_onnx`) says where the one score for each pixel of a segmentation
  is cut, in the model's own numbers: a pixel is the foreground at that score or more. Left out, it is found from the
  model, as above. It is refused, with `param="threshold"`, for a mask of several classes or of class numbers, for
  any other task, and when it is not a finite number (it settles a height and a width of whole-valued floats as
  one score for each pixel); `via="native"` refuses it as it refuses every argument only a
  conversion reads. It is the mask's `head.threshold`, not a classifier's: the class threshold a yes-or-no model is
  registered with stays `outputs={"threshold": ...}`.
- `task="segmentation"` and `task="object_detection"` send a PyTorch or Keras model through the conversion on
  `via="auto"`, with `input_shape` alone as with a `sample`: Kaimei's own readers read class scores and numbers,
  and a mask or boxes is read by a head only the conversion writes. `via="native"` refuses such a task, naming it.
- **Examples for text and image models** are in `examples/`: a scikit-learn text classifier
  (`text_sentiment.py`) and a small PyTorch segmenter (`image_segmentation.py`), each as a script and a
  notebook, from training to an upload, a prediction and an explanation or a mask.
- **Plans.** The Kaimei API limits by plan how many models your own agent runs and how many numbers one model
  graph keeps on Kaimei (`403 quota_exceeded`); every plan has every feature. `weights="summary"` sends a
  model's tensors as shapes, which are not counted.
- **The ONNX route, made stricter.** A model that draws at random, at its top level, in a branch or a loop or in a
  function the file defines, is checked against its seeded reading, not the unseeded original, so it is run, not
  drawn; its structure says what the check compares (`metadata["compared_with_original"]` is `false`, and
  `checked_against` ends ", with fixed seeds"). A NumPy sample is given to a PyTorch module in the module's own float
  type, on its device, and so are the numbers `preprocess=` normalizes with; converting a module leaves torch's
  random generators as they were. `on_unsupported="error"` refuses a graph that would only be drawn, and an upload
  the API says it only draws. On every route, a wrong `sample`, `inputs`, `output`, `tokenizer`, `labels` or
  `max_length` is an error that names the argument (`StructureError.param`), not a quiet fallback: a sample the
  model cannot be called with, an `inputs=` name the model does not have, a picture layout for an input that is not
  a picture, an output it does not give, a tokenizer for a model that takes no token ids, labels that are not a
  list of names (a classifier that gives one value for a row needs two, negative then positive), `feature_names`
  that do not fit the model on the conversion route. `inputs`, `tokenizer`, `mask_layout` and an `output` other
  than a module's first are read only from a model converted to ONNX: with any of
  them `via="auto"` converts the model first and refuses it when the conversion cannot be used, and `via="native"`
  refuses them. A conversion is never checked against itself: when nothing says which of a file's outputs is the
  library's answer, the reading is drawn. A scikit-learn tree fitted on rows with missing values, which `skl2onnx`
  never returns from converting, is refused before conversion wherever it sits in the model (inside a calibration,
  behind a target transform, in a stack or a vote); Kaimei's own readers read such trees. A Keras model that is not
  a table or a picture goes through ONNX instead of being refused for want of `feature_names` (without `tf2onnx`,
  or when its conversion cannot be used, it is uploaded as Kaimei's own readers draw it, as 0.11.0 uploaded it, with
  a warning that says why), and the warning for a model that is drawn gives the conversion's own reason, and what
  to pass.
- **Any model, through ONNX.** A model Kaimei has no reader of its own for (a recurrent network, a transformer, a
  model with several inputs, most scikit-learn estimators) is converted to ONNX in memory, on your machine, and
  uploaded as a graph of ONNX operators that the API runs. `models.upload`, `models.upload_structure` and
  `extract_structure` take `via="auto" | "native" | "onnx"` to pick the route, `sample=` (an example input),
  `inputs=` (what an input is), `output=` (which output is the prediction) and `labels=`. An ONNX model itself
  (a `.onnx` path, its bytes or an `onnx.ModelProto`) can be given in place of the estimator, and is read as a
  graph whatever `via` says; one that defines its own functions is read too (they are expanded with
  `onnx.inliner`, and an `onnx` package too old to have it is refused, saying to install a newer one). `task` may
  also be `text_generation`, `generation` or `embedding`. `labels=` takes a list, a tuple, a NumPy array or a pandas
  Index (`labels=encoder.classes_`) or a generator, which is read once and kept; `inputs=` names inputs of the model as converted (for a module whose `forward`
  takes more values than it names, `input_0`, `input_1`, ...); and `sample=` for a scikit-learn pipeline whose
  `ColumnTransformer` picks its columns by name is a table that holds those columns. `upload(model, X=frame)` with a
  frame of another width than the model was fitted on says so of `X` (`param="X"`), not of a `feature_names` the
  caller never gave.
- **An encoder-decoder kept as an ONNX file is read with its decoder's input hinted `{"generated": True}`.**
  `from_onnx(file, inputs={"decoder_input_ids": {"generated": True}})` (and `import_onnx`) writes a feature of token
  ids marked `generated` with the decoder's first token as its `baseline`, the input bound to that feature's ids
  (`{"feature": "decoder", "part": "ids", "pad_id": 0}`), the structure's `decoder` (`feature`, `start_token`,
  `eos_token`, `pad_token`, `max_length`), and the task `sequence_to_sequence`. The hint may also say `feature` (the
  feature's name, `decoder` by default), `pad_id` (what fills the input after the last token written, 0 by default;
  `None` is the same as leaving it out), `start_token` (the first id of the input's sample when one is given, else
  the padding token) and `eos_token`. The decoder's mask, positions or length that come with the ids by name
  (`decoder_attention_mask` beside `decoder_input_ids`) are bound to the same feature unless they have a hint of
  their own. A hint that cannot be written that way is refused, naming `inputs`: an input that does not hold whole
  numbers, two inputs marked generated, a task other than `sequence_to_sequence`, a hint that also says `value`,
  `zeros`, `features`, `layout` or a `part` other than `ids`, a `pad_id`, `start_token` or `eos_token` that is not a
  token id, a `feature` that is not a name or that another input reads, a sample that starts with something that is
  not a token id, and an input the model's output does not depend on. The importer's own check holds the model's
  scores at the last token the decoder's feature has in each row, which is where the API reads them.
- A classical classifier written without a ZipMap is read as its ZipMap file is: its classes are found behind a
  cast, in `classlabels_ints`, in the one column of a two-class linear SVM, and in the table a plain-step model's
  own label is looked up in at the argmax of its scores.
- Kernel models (SVC, SVR, NuSVC, and the wrappers around them) uploaded without rows are placed by their support
  vectors: far from zero they are read in double precision, or drawn when the converted model does not give the
  model's numbers there (ONNX keeps support vectors in single precision), and not called exact.
- **A classifier is exact only when its class is the model's own.** Besides its scores, the class Kaimei reports for
  a row is compared with the model's own `predict` on every row of the check, by the rule the API reports a class
  with. Where they differ, the reading that agrees is used: a `TunedThresholdClassifierCV` or
  `FixedThresholdClassifier` is uploaded with its own decision threshold (`outputs.threshold`, in the units Kaimei
  reports, and `outputs.positive_label` when its positive class is the first one), and a classifier whose `predict`
  goes by its scores where its probabilities tie (`SGDClassifier(loss="modified_huber")` of three classes or more,
  one-vs-rest wrappers of it) is read by its `decision_function`. When no reading agrees on every row, the model is
  uploaded drawn, not run, and its summary says on how many rows the class differs and why
  (`on_unsupported="error"` refuses it). Kaimei's own readers answer to the same rule, and so do scikit-learn
  pipelines and an ONNX file that gives its own label beside its probabilities. A model's own threshold
  (`TunedThresholdClassifierCV`, `FixedThresholdClassifier`) travels in its structure, and the API registers it
  under the model's own class names. `models.upload` sends in `outputs` only what the caller said, and
  `models.upload_structure` does not write a threshold to the model; both read the registered decision back and
  warn when it is not the model's own, naming the `client.models.update(...)` call that registers it. A
  `threshold` or `positive_label` passed in `outputs=` is the caller's own decision: it replaces the model's, is
  sent as it is, and warns with the number of checked rows on which it gives another class than `predict`. A
  decision registered by the customer is kept; a structure uploaded to an existing model changes the registered
  decision only when it was the default or the previous structure's.
- **An SVC or NuSVC says how its class is chosen.** With `probability=True` it is read by its probabilities when the
  most probable class of `predict_proba` is the class `predict()` gives on every row of the check; scikit-learn
  documents that the two can differ on other rows, so the upload warns and the structure says so in
  `metadata["predicted_class"]`. Where a check row disagrees, it is read by its `decision_function` instead, whose
  class is the model's own, and runs: on its own, behind a one-vs-rest wrapper and behind a table pipeline. Of more
  than two classes and read by its scores, Kaimei's class is the highest one-vs-rest score, where `predict()` counts
  one-vs-one votes and can differ where votes tie: the note and the warning say so, a model whose check holds a
  tie its scores settle another way is drawn, not run, and `break_ties=True` on the model makes the two agree. With
  `decision_function_shape="ovo"` and more than two classes its scores are for pairs of classes, not classes: it is
  refused, saying to set `decision_function_shape="ovr"` (which needs no refit).
- **A scikit-learn model read through ONNX is computed in the precision that gives its numbers.** Its check holds
  your rows as the doubles they are, and rows made up where the model's data lies (its first scaler's statistics,
  your rows, the points a nearest-neighbours or naive Bayes model keeps), not around zero. Single precision is kept
  only when it gives the model's numbers on those rows; otherwise the graph is in double precision
  (`metadata.model_precision`). Where nothing says where a column's values lie, a model that tells doubles apart is
  computed in doubles (a tree is not: scikit-learn rounds a tree's inputs itself), and the summary says to give
  rows of the table.
- The conversion is made by the framework's own converter: `torch.onnx.export` for a `torch.nn.Module`,
  `skl2onnx` for a scikit-learn estimator or pipeline, `tf2onnx` for Keras (the Keras route was not run for this
  release: TensorFlow was not installed where it was tested). `pip install "kaimei[onnx]"` installs `onnx`.
- The rows sent with a graph for the API to check itself on hold the outputs of the original library, not of
  the conversion. A conversion that differs from the library by more than 1e-4 of its outputs' size is marked
  `approximate`: drawn, not run. The conversion is compared on ONNX's own evaluator when ONNX Runtime has no kernel
  for one of its steps (a double-precision LSTM), and only then.
- **Text.** `tokenizer=` makes a token input a text feature. A Kaimei description (`char`, `word`, `wordpiece`,
  `bpe`), a Hugging Face WordPiece tokenizer (BERT) and a byte-level BPE tokenizer (GPT-2, RoBERTa) are written
  down as JSON and compared with the original on the sample texts of `kaimei.text.SAMPLE_TEXTS`:
  `kaimei.text.tokenizer_spec`. Any other
  tokenizer (SentencePiece, tiktoken, your own), and one that fails the comparison, is uploaded as its vocabulary
  alone (`"type": "pieces"`), with a warning: Kaimei names its tokens, and text is split by the model's agent.
- Special and added tokens written in a text (`[SEP]`, `<|endoftext|>`, tokens added to a tokenizer) are matched
  as Hugging Face matches them, and `_` and a final sigma are split and lowercased as Hugging Face does. These
  cases are among the sample texts the comparison uses, so a tokenizer that still differs on them is uploaded as
  `pieces`, with a warning. So are runs of 260 to 400 bytes with no break (Chinese and Japanese without
  punctuation, a long word, a long number, a run of symbols).
- **Words of any length are split as Hugging Face splits them.** Byte-pair merges are made as Hugging Face makes
  them, one at a time, lowest rank and leftmost first (a merge listed twice takes the rank of its last listing), so
  a run of more than 256 bytes with no break is split as the original splits it. Hugging Face's BPE options are
  followed: a character the vocabulary lacks is left out before merging, or is the unknown token (one for a run of
  them with `fuse_unk`), or its bytes with `byte_fallback`, and `ignore_merges` takes a word the vocabulary holds
  whole. A token added with `add_tokens` that a WordPiece vocabulary does not hold is only matched whole in a
  text, never as a piece of a word, and a tokenizer whose `max_input_chars_per_word` is above 256 splits longer
  words as it says. Tokenizers with these options are run by Kaimei, not uploaded as `pieces`. A tokenizer with two
  added tokens that are one text once normalized (`newtoken` and `NewToken` with an uncased normalizer) is
  uploaded as `pieces`, with a warning: Hugging Face takes one or the other from one run to the next. A byte-pair
  description whose merges name a piece its vocabulary lacks is refused when it is built, naming the piece, as
  Hugging Face refuses it.
- `max_length=` on `models.upload`, `models.upload_structure`, `extract_structure`, `graphs.via_onnx` and
  `graphs.from_onnx`: how many tokens a model that reads text is uploaded for. A graph's shapes are fixed, so
  without it the length of `sample` is the most tokens the model is ever given; a language model uploaded that
  way warns, because its prompt and what it writes share them. `max_length=` and `output=` take any whole number,
  NumPy's included.
- `models.generate` runs a language model's loop (or a model that draws), `models.tokenize` splits text or names
  token ids, `models.parameter_view` returns a picture of one tensor, and `models.agent_status` says whether a
  model's agent is connected.
- `models.generate(seeds=[...])` draws a model that answers in one pass once for each of up to 16 seeds, returned as
  `variations`, and `draw=[...]` names tensor or sequence features to draw from the seed although they have a
  baseline. `models.activations(channel_start=...)` sends the maps from that channel on, in their own order,
  `channels` at a time: how every head of an attention layer is read. Each is sent only when given.
- **Agents.** `kaimei.agent` runs a model where it lives: `serve(client, model, predict=...)` connects out to the
  API, waits for work and answers it. `predict_with(estimator)` makes the function for a scikit-learn-style
  estimator or a `torch.nn.Module`. With `module=` (PyTorch, forward hooks) or `onnx=` (the model's ONNX file,
  run with ONNX Runtime; `providers=` picks a GPU) the agent also answers trace and activations requests for a
  structure uploaded with `weights="summary"`. With `tokenizer=` it splits text for a model whose tokenizer
  Kaimei cannot run. From a shell: `python -m kaimei.agent --model ... --predict package.module:function`, or
  `--onnx model.onnx`. An agent answers an activations request the way the API does: the maps of a stage
  strongest first, or in their own order from `channel_start` (how the heads of an attention layer are asked
  for), up to 64 of them, and the heads of an attention layer ranked by their `focus`.
- An agent given an ONNX file, for a model with no structure on Kaimei, reads the prediction where the importer
  does: the first output of floats, and what a scikit-learn classifier gives beside its class (its probabilities,
  or its scores) in the tensor its `ZipMap` is made from, never its class label. A file none of whose outputs
  holds scores is refused, saying so. A converted classifier's probabilities are sent as they are. One without
  probabilities (an SVC, an `SGDClassifier`, a perceptron) gives scores, which are sent so that Kaimei reads from
  them the class the model predicts: one margin for two classes (its `decision_function`, positive for its second
  class), the scores for more, or the class itself where the scores do not give it (an SVC of three classes or more
  decides by votes, which can tie). What a file's numbers are is read from the steps that made them (a softmax or
  a sigmoid, a classifier's own transform, an SVC's probability model), not from its `ZipMap`.
- **An agent sends what a model computes, never the model.** The structure that names a model's steps comes from
  Kaimei, and an agent does not take its word for what a step is. A step is sent only where its values are
  computed from the row: a weight, a constant, and what is worked out from those alone (or from an input's shape
  alone, like the embedding of each position; or from an input that is given the same value for every row, like a
  text's segment ids) are the model's own numbers and are left out, whatever step the structure says they are.
  With `onnx=` that is read off the file's graph; with `module=` the run is followed one operation at a time
  (PyTorch 1.13 or later; before that, what the modules' hooks can tell). A prediction is read the same way: an
  output of a file that holds the model's own numbers is passed over for the first output of floats that is
  computed from the row. A step's values are laid out only as its tensor is: the shape the structure gives must be
  the tensor's own, and the order of its axes a true order of them. A step that is kept back is named, with the
  reason, under `withheld` in the answer to a trace or an activations job (at most 64 of them), and once in the
  agent's log; the job is answered all the same.
  A tensor a PyTorch module keeps as a plain attribute (not a parameter, not a buffer; itself, or in a list, tuple or
  dict that is the attribute) is the model's own from the start of every look, so a step that gives it, a part of
  it, or something computed from it alone is withheld at every look and not only at the look that made it; what the
  module is called with for the row is never taken for the model's own. A tensor kept deeper than that (in a module
  that is not registered, in a closure or a global, or in another kind of object) is not recognised unless the run
  computes it from parameters, buffers or such attributes, and before PyTorch 1.13 a tensor a module computes from
  its weights while called with the input is sent at the look that makes it.
- An agent started with the structure that was uploaded (`structure=`) goes by that structure: it says nothing of a
  step it does not hold, reads a file's steps from the tensors it names, and reads the prediction where its own
  head does, whichever step the structure on Kaimei points at.
- An agent leaves a PyTorch module as it found it: each part in its own training mode (a frozen backbone stays
  frozen), and no hook behind after a job that fails. `predict_with` gives a scikit-learn estimator fitted on
  named columns each where it was fitted to find it, whatever order `feature_names` names them in. An agent
  started with the uploaded structure feeds an input its fixed value; one a summary upload kept at home is refused
  with what to do (`weights="full"`, or `inputs={name: lambda row: ...}`, the input named as the model file names
  it). `Agent(inputs=...)` beside `structure=` feeds each input it names as it says and every other input as the
  structure says; a name in `inputs=` that is not an input of the model's file is a `ValueError` when the agent is
  made, which lists the inputs there are, and the structure's own name for one (`in.state`) is answered with the name
  that was meant (`state`).
- **An agent runs an encoder-decoder.** A `sequence_to_sequence` model's prediction is the scores of the next token,
  read after the last token its decoder has written so far (the feature marked `generated`, which Kaimei sends with
  every row and which starts as the decoder's first token). With `onnx=`, the inputs the file names for its decoder
  (`decoder_input_ids`, `decoder_attention_mask`) read that feature without being told, and an agent started with
  `structure=` feeds them the same way, also where the structure describes the input as `{"generated": true}`.
  `POST /generate` then writes through the agent as it does on Kaimei. Without `structure=`, with several text
  features (an encoder-decoder's two), whole-number inputs are bound by themselves only when `decoder` in a name
  sends it to the feature marked `generated` and every other name to the one text given, and each text's token ids
  are read by exactly one input (`input_ids`, `attention_mask`, `decoder_input_ids`, `decoder_attention_mask`);
  any other naming (`src` and `tgt`, two inputs for one text, one input for two texts) is refused, saying to give
  `inputs={...}`. A model whose only text feature is the one marked `generated` is bound whatever its input is
  called. A model on Kaimei without such a feature is refused with the reason, for each job. With `module=`, give
  `prepare=` a function from a row (`row["text"]`, `row["decoder"]`) to the module's arguments.
- An ONNX file whose nodes list inputs left out at their end under empty names (an opset-20 `DFT`), at its top
  level, in a branch, in a loop's body or in a function of its own, is read and run without them, which means the
  same: ONNX Runtime 1.32 stops the process on some such nodes.
- An agent says what its model's own output is (`"link"` in its answer), so a model with no structure on Kaimei
  is not squashed twice: `"none"` for probabilities (a module or file that ends in a softmax or a sigmoid, a
  converted classifier's probabilities, `predict_with` around `predict_proba`), `"softmax"` for a log-softmax. A
  module is read with `torch.fx`, or by which of its modules gave the output; a file by the node that makes the
  prediction; where neither says, by whether its rows are probabilities. Scores say nothing, and Kaimei reads
  them through a softmax or a sigmoid.
- A look inside a large model through `onnx=` reads its steps in several runs, and keeps a session of ONNX
  Runtime for each so the next look makes none of them again, as `session_memory=` allows
  (`Agent(session_memory=...)` in bytes, `python -m kaimei.agent --session-memory MIB`; 2 GiB unless said; a
  session holds about three copies of the weights). A look larger than the memory makes again only the sessions
  that do not fit, and the session that gives a prediction is always kept. Those sessions do without ONNX
  Runtime's memory arena, so each keeps its weights and not what its last run took. A run reads at most 256 tensors
  and 16,777,216 values (`kaimei.agent.FETCH_STEPS`, `FETCH_VALUES`), counted by what the file says each tensor
  holds (its declared shape with the row's sizes, a shape asked for in a run of its own where the file does not
  say, and what it was found to hold once read), never by the size the structure on Kaimei gives a step, and a
  tensor that several steps name is read once. A run's values are made into the frames or stages that are sent and
  let go before the next run, so a look inside a large model takes the memory of one run.
- An answer larger than the API takes (4,000,000 values or 128 MiB of JSON) is not sent: the job fails, saying how
  large it was. An answer the API refuses for any other reason than its job being gone, or that is lost on the
  way, is followed by a short failure, so whoever asked is told why rather than waiting out the timeout. Nothing
  a job does ends an agent: an answer that cannot be written as JSON is followed by a short failure too, a failure
  that cannot be sent either is said in the log, and the agent goes on to the next job. A step's statistics are
  numbers whatever its values: a length or a sum too large for a float is sent as the largest float there is.
  What cannot be written as JSON is said for what it is: "Infinity can't be sent" only for a number that is not
  finite; text that holds half of a surrogate pair is said to hold a character that cannot be written as UTF-8,
  with the character and what comes just before it; a list or a dict that holds itself is said to.
- `python -m kaimei.agent` stops on SIGTERM as on Ctrl+C and closes its session, so a stopped container leaves no
  session for jobs to be queued to.
- An agent holds a request for work open for less than the client's own timeout (7.5 seconds for
  `Kaimei(timeout=10)`), and pauses to the end of a second after one that comes back empty sooner (`wait=0`). An
  agent whose session the API has lost opens another at once the first time, and after a growing pause (1, 2, 4
  seconds and so on, up to 30) when the new one is lost too.
- `save_onnx=` on `models.upload` and `models.upload_structure` writes the converted ONNX model to a path, for
  the model's agent to run. `Extraction.onnx` holds the same bytes, and `graphs.to_onnx` converts a model again.
  The file is written only once the structure read from it is the model's: not when an older API refuses the
  graph and the model is uploaded as Kaimei's own readers read it (the warning says no file was written), not when
  `on_unsupported="error"` deletes the model, and not when `upload_structure` is refused. A path that cannot be
  written (a folder that is not there) is refused before anything is uploaded, with `param="save_onnx"`; a file that
  exists and can be written is accepted whatever its folder allows; a write
  that fails after the upload is a warning that names the model, whose id is returned.
- `weights="summary"` works for graphs, and sends what it always has: shapes and statistics, and no learned
  number however few (a bias, a class's mean, a converted classifier's coefficients). A graph keeps only the
  whole numbers its shapes are worked out from (shapes, axes, indices), the scales of a resize, and its
  operators' single settings (an epsilon, a slope). A float value an input is always given
  (`inputs={name: {"value": array}}`) is sent as its shape too, and the reading is drawn. Statistics are
  computed in pieces, without a double-precision copy of a tensor. The API no longer limits the size of a
  tensor or the number of parameters of a model sent this way.
- `weights="pictures"` is a summary that also sends a small picture of each float tensor of 256 values or more
  (`view`: at most 64 by 64 cells, the strongest value of each block, at 8 bits), which the dashboard draws
  where the weights would be. It is asked for, never the default: a tensor of 64 by 64 values or fewer is all
  there in its picture. A tensor is pictured as a matrix, its first axis down and the rest across, with leading
  axes of one left out (a recurrent layer's `(1, 4 × units, inputs)` weights are `4 × units` rows), as the API
  draws one it holds.
- Binary tensors upload in their own number type (`f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`,
  `bool`); `upload_tensors(dtype=...)` still sets one for all.

### Changed

- `pip install "kaimei[onnx]"` also installs `onnxscript` (0.7.2 or later), which PyTorch's newer exporter needs:
  Hugging Face language models and encoders only convert through it. It needs `onnx` 1.17 or later, which the
  extra now asks for.
- Estimators that were refused (`SVC` with a non-linear kernel, k-nearest neighbours, naive Bayes,
  `IsolationForest`, pipelines with `PCA` or `PolynomialFeatures`) and ones that were drawn (`SVC` and `NuSVC`
  with `probability=True`) are uploaded through ONNX and run, when `skl2onnx` is installed (such an SVC says how
  the class Kaimei gives a row is chosen: see "An SVC or NuSVC says how its class is chosen" above). Without it they
  are refused or drawn as before, and the error says what to install. `via="native"` keeps the old behaviour.
- A PyTorch module the native reader could only draw (attention, exact GELU, a reshape) is converted through
  ONNX when `onnx` is installed. One it could not call (token ids, several inputs) is converted first.
- Reference rows (`X=`) are stored for tables and images. For a model that reads text, a series or a tensor
  they are left out, with a warning.
- `predict_with` answers for a classifier without `predict_proba` with its `decision_function`, marked as the
  model's own scores, where it used to answer with class labels.
- Three modules are copies of the API's, kept byte for byte: `kaimei/onnx_reader.py`, `kaimei/onnx_graph.py`
  and `kaimei/tokenizers.py`.

## 0.11.0 — 2026-11-25

Needs Kaimei API 1.2.0 for `organizations.confirm_join_domain`. Against 1.1.0, everything 0.10.0 did works as
before. Against 1.2.0 the calls are the same, and four answers change because the API changed them:
`organization.enable_join_domain()` emails a link and turns nothing on until it is opened, a transfer drops
endpoint credentials and can be refused with `403 quota_exceeded` for this month's usage, and `auth.signup`
without `accept_terms=True` is refused with `400 terms_not_accepted`.

### Added

- `auth.signup(..., accept_terms=True)` says the person agreed to the Terms of Use, which Kaimei API 1.2.0 requires
  to make an account and records with the Terms' version (`user.terms_version`, `user.terms_accepted_at`). Clients
  0.10.0 and older cannot send it, so their `auth.signup` is refused by 1.2.0. `auth.reset_password(...,
  accept_terms=True)` is for an account that was never confirmed, which the reset makes yours: 1.2.0 answers
  `400 terms_not_accepted` until it is passed, and the same link works then.
- `organizations.confirm_join_domain(token)` turns joining by email domain on with the token from the link the API
  now emails the admin who asks. Needs Kaimei API 1.2.0, where `organization.enable_join_domain()`
  emails that link and answers with `enabled` false and `pending` set, instead of turning it on at once.

### Changed

- The docstrings of `models.transfer` and `projects.transfer` say that, with Kaimei API 1.2.0, endpoint
  credentials are removed rather than moved and the destination needs room for this month's usage. Nothing in the client
  changes for either: both come back in the result or as `403 quota_exceeded`.

## 0.10.0 — 2026-11-21

The new calls need Kaimei API 1.1.0. Everything 0.9.2 did works as before, against 1.0.0 or 1.1.0.

### Added

- **Organizations.** An account has a personal workspace and can belong to organizations. `organizations.list()` and
  `organizations.create(name=...)` list and create them, and `auth.switch("acme")` makes the client work in one
  (`"personal"` goes back). These need a session from `auth.login`; an API key keeps working in the organization it
  was created in.
- `organization.members()`, `organization.update_member(user_id, role=...)` and `organization.remove_member(user_id)`
  manage who is in the organization: `owner`, `admin`, `member` or `viewer`.
- `organization.invite(email=..., role=...)` returns the invitation with its `invite_url`;
  `organization.invitations()`, `organization.resend_invitation(id)` and `organization.revoke_invitation(id)` manage
  them. The person invited uses `invitations.list()`, `invitations.accept(...)` and `invitations.decline(...)`.
- `models.transfer(model, organization=...)` and `projects.transfer(project, organization=...)` move work, with its
  history, to another organization. `models.fork(model)` and `projects.fork(project)` copy it, to your personal
  workspace unless `organization` says otherwise.
- `organization.delete(confirm=slug)` deletes a team organization.
- **Joining by email domain.** `organization.join_domain()`, `organization.enable_join_domain(role=...)` and
  `organization.disable_join_domain()` let anyone at an admin's own email domain join. Someone at that domain uses
  `organizations.joinable()`, `organizations.join(organization_id)`, which emails them a confirmation link, and
  `organizations.confirm_join(token)` with the token from that link.

## 0.9.2 — 2026-11-08

A 0.0.1 bump from 0.9.1. The calls are unchanged. Structure documents may include `verification`.

## 0.9.1 — 2026-11-08

Needs the API release with password reset: deploy that first.

### Added

- `auth.forgot_password` and `auth.reset_password` request a reset link and set a new password from it.

## 0.9.0 — 2026-11-08

Needs the API release with outcomes and the `performance` analysis: deploy that first.

### Added

- **Outcomes and performance over time.** `predictions.create(actual=...)` logs what really happened with the
  prediction; `predictions.log_actual(id, actual)` and `predictions.log_actuals(...)` attach it later, from a dict,
  a list of `{"prediction_id", "actual"}` items or a pandas Series indexed by prediction id. Batches of any size are
  sent 1,000 at a time; NaN and missing values are skipped, and an explicit `None` in a list clears an outcome.
  A classifier's outcome is its class name, its position, or `True`/`False` for a binary model; a regression's is a
  number.
- `models.performance(model, window="week", days=90)` measures the model against outcomes: AUC, accuracy, precision,
  recall, F1, Brier score and calibration for binary models; accuracy, macro F1, per-class numbers and a confusion
  matrix for multiclass; MAE, RMSE, bias and R² for regressions; overall and per day, week or month.
  `display.rows` turns its series into a table.
- `analyses.create(type="performance")` compares the newest outcomes with the ones before them (or with the
  labelled reference rows, `options={"baseline": "reference_data"}`) and raises a `high` insight when the drop is
  larger than chance explains. Full analyses include it.
- `predictions.list(has_actual=True)` lists only predictions with an outcome. Notebook cards show the outcome and the
  performance insight.

## 0.8.0 — 2026-11-07

Needs the API release that compiles `opaque` layers: deploy that first.

### Changed

- **Every PyTorch and Keras model uploads and is drawn.** A layer or operation Kaimei cannot replay (attention,
  `matmul`, a reshape to another rank, a channel shuffle, a module of your own) becomes an `opaque` step in the spec:
  the framework's own name for it, its attributes, its output shape and its tensors, which the dashboard draws and
  lists like any other layer. The model is then drawn, not run, and its summary says which steps and why. Before,
  such models were refused with `unsupported_layers`; `on_unsupported="error"` on `upload`, `upload_structure`,
  `extract_structure` and `estimator_to_spec` keeps that behaviour.
- Dense PyTorch models that `torch.fx` can trace keep their op-by-op reading even when a step is drawn, so a
  residual add or a functional activation next to it still runs the way the code says.
- `kaimei.structure.drawn_steps(spec)` lists the steps of a spec that are drawn but not replayed.
- **Every spec says where it came from.** `metadata` on the spec records the model's class, the framework and
  its version, the Python version, this client's version and when it was read; `extract_structure(metadata=...)`
  and `upload_structure(metadata=...)` add your own entries. The API keeps it on the structure document, and the
  dashboard shows it under Details. (`upload(metadata=...)` is still the model's own metadata.)
- Parameters in the structure document carry their stored precision (`dtype`: `f16`, `f32` or `f64`).

## 0.7.2 — 2026-11-07

Needs the API release that understands flat forests and composites: deploy that first.

### Changed

- **Uploads are checked more strictly, and some models are drawn instead of run.** A model is sent with a few rows and
  its own library's answers to them, and the API scores those rows with the engine that will run it. If they don't
  match, the model is drawn, not run. The rows aren't stored, and `weights="summary"` doesn't send them. That covers
  PyTorch and Keras models, convolutional ones included. `SVC` and `NuSVC` with `probability=True`, whose labels can
  disagree with their probabilities, and any subclass with its own `predict` are drawn too. **Models already uploaded
  keep their old label until they are re-uploaded**, because the server never holds the model itself to check it.
- Dense PyTorch networks are read op by op, as convolutional ones already were, so `F.relu`, `torch.sigmoid` and a skip
  connection from the inputs run correctly rather than being missed. Modules `torch.fx` can't trace still go through
  forward hooks and are drawn.
- `nn.GELU()`, `F.gelu` and Keras `gelu` are drawn, not run: they default to the exact form, and Kaimei runs the tanh
  form, which is too close for a check on convolutional models to tell apart. `approximate="tanh"` runs.
- Reference rows go up in pieces of about 32 MiB, so a large `X` no longer hits the API's limit on one request.
- `upload(X=...)` also checks the model on your rows, not only on rows around zero and its splits.
- Summaries and cards say what a base value is, and call a model's outputs scores when it has no probabilities.
- A plot of a very wide model has a row for what wasn't attributed to one input, so its last bar is the prediction.
- `upload` warns when reference rows were refused, instead of leaving them out silently.

### Fixed

- **Models could upload as something they aren't, and report wrong numbers as exact.** The extractor
  walked into fitted children and kept the first one that looked exact. A forest over 30 trees or
  4,000 nodes, sklearn's default `RandomForestClassifier()` included, ran as its first tree. A
  `VotingClassifier`, `BaggingClassifier` or a vendor's list of models ran as its first member, and
  `CalibratedClassifierCV` and `TransformedTargetRegressor` ran without the step they add. **Re-upload
  any model affected by this**: a stored spec keeps its old numbers.
- AdaBoost, and bagging over a subset of columns, no longer run as a plain average of their trees.
- Scaler pipelines apply every scaler as sklearn does. Only the last of several scalers was kept, and
  `StandardScaler(with_mean=False)` or `with_std=False` used the wrong centre or scale.
- Single trees and small forests compare inputs in single precision as sklearn does, which decides a
  row that sits exactly on a split.
- `XGBRegressor(objective="reg:logistic")` and `LGBMRegressor(objective="cross_entropy")` reported one
  minus their output. Bare XGBoost and LightGBM boosters now carry class labels and go through the check.
- `XGBClassifier(missing=0)`, or any other marker for missing values, now takes the branch XGBoost does
  for that value. Before, it scored those rows as ordinary numbers.
- A `CatBoostRegressor` no longer uploads as a multiclass classifier, whichever way its trees grow.
- A PyTorch layer called twice no longer fails with `Duplicate structure node`.
- `predictions.retrieve` encodes the id, so one holding `?` or `#` reads back.
- Models fitted on data with missing values can be uploaded. sklearn gives them infinite thresholds, which
  JSON can't carry, and those thresholds also made the check skip them.
- A model with several regression outputs is drawn but not run. It used to report its first output as the
  model's only output, as exact.
- `upload(X=...)`, or any call whose body held `NaN`, raised a raw `ValueError`. `NaN` now travels as `null`,
  which is how JSON writes a missing value.
- `predict`, `explanations.create` and `trace_structure` raise `ValueError` for a `NaN` on a feature that can't take
  a gap, after one extra request to read the model. As `null` it would have used the feature's baseline.
  Infinity in a request now says what's wrong.
- **`weights="summary"` sent real numbers.** Trees, small forests, starting scores, per-tree weights, calibration
  curves and scaler statistics all went up as they were, and the statistics of a small tensor gave away its values.
  Now a summary holds shapes, feature importances or how often each feature is split on, and statistics only for
  layers of 256 values or more. Those statistics still include a layer's largest and smallest value.
- A pipeline's `SimpleImputer` is no longer read as handling gaps, which was wrong for some rows. Its fill values
  become the features' baselines instead, so a `null` gets what the imputer would give it, and reference rows with gaps
  are kept. A `SimpleImputer(missing_values=-999)` raises `StructureError`: it replaces real numbers, so reading it as
  nothing was wrong for those rows, and the check didn't try them.
- PyTorch and Keras models ran as exact with wrong numbers when `forward()` held logic that wasn't read. They now run
  correctly when it can be read, and are drawn when it can't.
- A PyTorch classifier that ends in logits gets its sigmoid or softmax, as image models already did.
- A scaler in front of any tree model is exact on rows sitting on a split. A threshold is now moved into the input's
  units by searching with the scaler's own arithmetic and the library's own comparison, not by multiplying it back out.
  This also lets a scaler precede XGBoost, LightGBM and CatBoost.
- Multiclass `SGDClassifier(loss="log_loss")` and `OneVsRestClassifier` over anything with a margin (logistic regression,
  `LinearSVC`, `RidgeClassifier`, `Perceptron`, `LinearDiscriminantAnalysis`, a linear `SVC`, or those behind a scaler)
  decide by their margins, as scikit-learn does, where two classes' probabilities round to the same number.
- A model only counts as matching if it was actually compared. One whose `predict` needs a table is now asked with one,
  and one that refuses to answer at all is drawn, not run. It used to be called exact without being compared.
- A subclass with its own `predict`, or your own class, inside a vote, a bag or a stack is drawn, not run. Only the
  top of a model was looked at before.
- A hard vote or `OneVsRestClassifier` over `SVC(probability=True)` is drawn, not run: its members decide by their
  margins, and what's read of them is their probabilities.
- An imputer behind a scaler gives a `null` the fill value in the input's units, not the scaler's.
- A scaler in front of a stack with `passthrough=True` also reaches the features it passes through.
- XGBoost's `missing` marker behind a scaler is drawn, not run, because no one input number stands for it.
- A classifier with several outputs of different sizes is drawn, not run. It was read as two classes with odd names.
- A `PoissonRegressor` and the other log-link models run where their library overflows to infinity.
- Members that don't fit together, such as a multiclass `SVC` in a bag or a vote, are drawn. `upload` raised a raw
  `ValueError`.
- A pipeline inside a pipeline's preprocessing is read, and two scalers in a row no longer clash on a one-feature model.
- `TransformedTargetRegressor` runs with `MinMaxScaler` and `RobustScaler` as well as `StandardScaler`.
- A two-output PyTorch or Keras classifier with `task="binary_classification"`, and a Keras `Sequential` that ends in
  logits, run. So do Keras models with a softmax after `Add` or `Concatenate`, or dropout after their softmax.
- A layer or a reader's own step can share its name with a feature. `upload` used to refuse the model.
- LightGBM fitted on a table with spaces in its column names keeps them, and `X` is read before anything is created, so
  an `X` that doesn't fit leaves no model behind.
- `feature_names` that moves any name the estimator was fitted on is refused, even if others are new.
- A PyTorch model whose `forward()` feeds its layers in a way that can't be read gets a clear `unsupported_layers` error,
  not the API's complaint about a weight's shape.
- A Keras model ending in `Dense(n)` then `Softmax()` or `Activation("softmax")` is read as a classifier. It was refused.
- A skip connection from a Keras network's raw inputs runs. It was refused with an unhelpful message.
- CatBoost, LightGBM and XGBoost models fitted on a table keep its column names when `feature_names` isn't passed. The
  names a library makes up for a bare array (`0`, `Column_0`) are no longer taken for real ones, so `X`'s columns name
  the model.
- A multiclass `SVC` is drawn, not run. It votes across pairs of classes, which a linear reading can't do.
- `feature_names` in a different order from the estimator's `feature_names_in_` raises `StructureError`.
- The check against the estimator scales with the size of the outputs. Outputs around `1e-7` used to match anything.
- A `LightGBM` or `XGBoost` subclass keeps its class labels and its checks. The wrong number of `feature_names` is
  a `StructureError` for every kind of model. For trees it used to be accepted, or a raw `IndexError`.

### Added

- Tree models of any size now run: forests, gradient boosting, histogram boosting, AdaBoost, bagging,
  XGBoost, LightGBM and CatBoost. Large ones travel as flat binary arrays in full precision.
- Models made of other models now run: voting (soft, hard, weighted), stacking, calibration, bagging,
  one-vs-rest, and `TransformedTargetRegressor` with a scaler or a NumPy inverse.
- `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` with a log link, a binary linear
  `SVC(probability=True)` and multiclass `SGDClassifier(loss="log_loss")` run exactly.
- `RobustScaler`, `MaxAbsScaler`, any `MinMaxScaler` range and `SimpleImputer` work in pipelines, and a
  scaler can precede any tree model.
- Every scikit-learn, XGBoost, LightGBM and CatBoost upload is checked against the estimator's own output,
  including rows exactly on its splits and rows holding zeros. A spec that doesn't reproduce it is drawn but
  not run, with a warning. Keras and PyTorch models are scored against their own outputs too.
- XGBoost's `dart` booster runs, and AdaBoost runs inside a calibrator or a stack.
- **Rows with missing values run, on models that handle them.** The check also scores rows with gaps. Where the
  library accepts them and the reading matches, numeric features get `allow_missing` and a row can send `null`.
  Shapley methods explain such a row; LIME and counterfactuals return `missing_value_not_supported`.
- `LinearSVC`, `RidgeClassifier` and others without `predict_proba` are marked `uncalibrated`, because
  their 0..1 outputs are scores rather than probabilities.

### Changed

- `client.models.upload_tensors` sends double-precision arrays as `f64` unless you pass `dtype`.

### Changed in the API this release needs

These change results for models you already have, and they don't depend on the client version:

- **Rows with a missing value now count.** Analyses and reference averages used to leave them out, and KernelSHAP
  explained a missing value as if it were present. Stored insights and explanations of those models may differ.
- `inference.url` can't be a loopback, private-network or cloud metadata address (`invalid_inference_url`), and moving
  it to another host needs its credentials sent again (`inference_credentials_required`).
- A reply from an endpoint is limited to 64 MB and to `timeout_s` in all, and an organization can have 12 calls to
  endpoints in progress at once (`429 inference_busy`). Analyses wait for a free call instead of failing.
- An organization's analyses take turns, leaving a worker for other organizations, and a job still in line isn't
  reported as lost.
- Replacing a model's structure updates `allow_missing` to what the new structure does. Class labels follow the order
  of the outputs. A `target_class` analysis scores accuracy for that class, and rows with no label no longer count.
- A binary `target_class` analysis now names that class. It used to name the positive class while reporting the other
  class's share.
- A forest with no rule for missing values refuses a missing value (`missing_value_not_supported`), even when a feature
  declares `allow_missing`.
- A link from outside the dashboard asks before signing you in, and a confirmation link works once.
- JSON nested more than 64 levels deep is refused (`too_deeply_nested`), an image feature can hold at most 8,000,000
  values, and an address over 254 characters isn't an email. A `prediction_id` holding a slash can now be read back.
- An email address that signed up with a password and never confirmed it loses that password when its real owner
  signs in with Google or GitHub. Otherwise whoever signed up first would get in to the owner's account.
- **A multiclass explanation with no `target` now explains the predicted class.** It used to explain the
  first class while naming the predicted one, so stored explanations of those models may differ.
- A multiclass analysis says which class it covers, and takes `options.target_class`.
- Explaining a row with nothing to compare it with returns `no_reference_population`. It used to return
  all-zero attributions.
- `NaN` or infinity as an input is refused. It used to return a prediction, often of 0.
- An output that isn't a finite number returns `non_finite_output`. It used to be reported as 0.
- A network with several regression outputs also returns `values`, all of them. `value` is still the first,
  and explanations explain that one.
- Analyses run in a small pool of their own, so a burst waits in line (`status: queued`) rather than making other
  requests fail. A job lost to a restart reports `failed` after an hour rather than `running` forever. A busy
  database returns `503 database_busy` with `Retry-After`.

## 0.7.1 — 2026-11-01

### Fixed

- Notebook cards keep their layout on GitHub and in other viewers that drop `<style>` blocks from
  HTML outputs. Each element now repeats its resting style inline, reading the theme's colours with
  the light theme as the fallback, and the logo and charts carry size attributes. Before, GitHub
  showed the logo as wide as the page and the rest unstyled. The stylesheet still drives the dark
  theme, hover and open sections in Jupyter, VS Code and Colab.
- GitHub's viewer also removes `<details>` and `<summary>`, which unfolded every section. The
  sections still read well open, except the JSON dumps: those are now hidden where the stylesheet
  is missing and shown, folded, everywhere else.
- The what-if steps of an explanation card name the probability they report: `→ P(survived) 33.9%
  · died` rather than `→ 33.9% died`, which read as the probability of dying. The before/after line
  says `P(survived) 7.4% → 33.9%` instead of `Output`.
- List cards (pages of models, explanations, predictions or insights) are wider, so every column
  fits; before, the last columns of a model list scrolled out of view.
- mypy passes again on the card renderer (CI).

## 0.7.0 — 2026-10-31

### Added

- `client.models.activations(model, input=...)`: what every layer computed for one input
  (`POST /v1/models/{id}/structure/activations`). Every stage reports its intensity, active share and
  strongest channels or units; the detailed stages carry each channel's energy, the maps of the
  strongest channels, a saliency map and, for narrow vectors, every value. `nodes=` picks the stages
  to detail, `channels=` how many maps, `cam=[layer]` adds class attention (Grad-CAM) at a layer
  toward `target=`, within the API's time budget.
- `kaimei.activations`: `decode` for the byte-packed maps, `stage`, `feature_maps`, `saliency`,
  `unit_values`, `class_attention`, and `plot_activations` (matplotlib), which draws the dashboard's
  Sight strip: the picture, then each detailed feature map's saliency over it and its strongest
  channel maps.
- A record that ends a Jupyter cell (an explanation, an insight, a model, a prediction, an
  analysis, a page of any of them, or anything else with an `object` type) displays as a Kaimei
  card: the mark, the dashboard's colours and type, and the dashboard's numbers formatted by the
  same rules (`fmtSigned`, `pct`, `outputParts`, `compact`, ... ported name for name).
  - An explanation shows the prediction (`died · P(died) = 92.6%`), the method, base value, model
    calls and latency, a one-sentence summary, the diverging attribution bars with the input value
    beside each feature (hover for six decimals; the rest of the features fold away), the class
    probabilities, the counterfactual step by step, and the input and the JSON behind collapsible
    sections. An image explanation shows the photo, the evidence heatmap and the focus (NumPy and
    Pillow, when installed), the evidence check and its numbered regions.
  - An insight shows its severity and score, the summary, and the payload as the dashboard draws
    it: importance and interaction shares, drift by status with the PSI, behaviour curves, the
    output histogram around the threshold (hover a bin), robustness, the learned rules.
  - A model shows its task, framework, status, counts, outputs and features; a prediction its
    output and input; an analysis its status, progress and timings; a page a table of its items
    with the total and the window shown.
  - The cards are HTML and CSS only (no script), render in JupyterLab, classic Jupyter, VS Code and
    Colab, follow the host's dark theme, and link to the record in the dashboard
    (`KAIMEI_DASHBOARD_URL` for a self-hosted one). `kaimei.display.html(record)` returns the HTML.
- The plain data: `record.to_json()`; `record.to_frame()` (pandas) and `kaimei.display.rows(record)`
  (no pandas) give the record's main table: one row per feature of an explanation (the API's
  `top_features` rows as sent, then the remaining attributions ranked with `share` by the same
  formula; regions for an image explanation), an insight payload's table (importance, drift,
  interactions, fragile features, rules, one row per grid point of each behaviour curve, histogram
  bins), a model's features, one flattened row per item of a page (`output.label`,
  `attributions.Age`, ...).
- `kaimei.set_display("raw")` makes cells show the JSON instead (a collapsible tree in JupyterLab);
  `kaimei.display.using("raw")` does it for one block; `KAIMEI_DISPLAY=raw` sets the default.
  Nested values such as `explanation.output` keep their plain repr in every mode.

## 0.6.3 — 2026-10-25

### Fixed

- `kaimei.tabular` chart titles use bold, which every matplotlib install has. Newer matplotlib
  logged `findfont: Failed to find font weight semibold for DejaVu Sans` under each chart.

### Documentation

- README: an Examples section (the [Titanic example](https://github.com/KaimeiAI/titanic-kaimei-example), the scikit-learn and image examples); links to [kaimei.ai](https://kaimei.ai) and the [API reference](https://kaimei.ai/docs) instead of a repository that isn't public.
- Package metadata: the "API" link points at the public API reference, plus an "Examples" link.

## 0.6.2 — 2026-10-25

### Fixed

- `kaimei.tabular.plot_insight` on a drift insight: the "moderate" and "significant" labels of the
  PSI reference lines overlapped when one input had drifted far (the lines sit close together on a
  wide axis). Each label now sits on the outer side of its line.
- `plot_explanation` for regression-style outputs (for example TreeSHAP values in log-odds) labels
  every bar with one shared precision, three significant figures of the largest, so a tiny
  attribution reads `+0.00` rather than `+0.000554`.

## 0.6.1 — 2026-10-25

### Fixed

- A request that timed out after reaching the server was retried whatever its method. A slow
  `POST` could therefore run twice or more: a batch of predictions that outlived the timeout was
  logged four times. Requests are now retried after a network error only when they are safe to
  repeat (`GET`, `HEAD`, `OPTIONS`) or never reached the server (connection refused or timed out).
  Otherwise `APIConnectionError` says the request was not retried because the server may already
  have applied it. Retries after 429, and `GET` retries after 502/503/504, are unchanged.

## 0.6.0 — 2026-10-25

Tabular explanations and model insights, drawn and worded the way the dashboard shows them.

### Added

- `kaimei.tabular`, for models whose inputs are table rows:
  - `explanation_summary(explanation, model=None)` states the prediction against the model's average
    and names the inputs that moved it most, in percentage points of `P(target)` for classifiers and
    in output units for regressions.
  - `plot_explanation(explanation, model=None, max_features=10)` draws a waterfall from the average
    prediction (the base value) to this one. Bars push toward the explained class (blue) or away
    (pink); the smallest attributions fold into one row; a hairline marks a binary model's threshold.
  - `counterfactual_summary` and `plot_counterfactual` show the counterfactual search step by step:
    the changes it made, the prediction after each, and whether the decision flipped. Requires an
    explanation created with `options={"counterfactual": True}`.
  - `compare_explanations([e1, e2], ["Model A", "Model B"])` puts one input's attributions from up
    to three models side by side.
  - `plot_insight(insight)` / `plot_insights(page)` draw every insight type an analysis produces:
    - importance, interactions and robustness as ranked bars;
    - behaviour curves as small multiples with individual-row curves;
    - the prediction histogram around the threshold;
    - drift, with PSI coloured by status and labelled in words;
    - the learned rules.
  - Passing `model=` (from `client.models.retrieve`) adds feature units (`Age = 22 years`) and names
    the output (`P(survived)`).
  - The text helpers need only the client. The plots need matplotlib, which `import kaimei.tabular`
    does not load.

### Changed

- A nested object without an `object` type, such as `explanation.output` or an item of
  `explanation.top_features`, now prints as its data instead of `<Kaimei object ...>`.

## 0.5.0 — 2026-10-25

Image explanations are drawn the way the dashboard draws them, and read like a sentence.

The summary, the numbered regions, the focus region, the evidence check and blurred hiding read
fields that the matching Kaimei API release adds to image explanations (`options["summary"]`,
`regions[].anchor`, `focus`, `evidence`, `background`). Against an older API the helpers still
work from the heatmap alone: the summary is built from it, regions follow the patch grid, and the
evidence pictures are marked "not measured".

### Added

- `kaimei.images.render_explanation(image, explanation, mode="heatmap" | "spotlight" | "original")`
  renders an explanation record from its `method`, `heatmap` and `focus`, in the dashboard's
  palette: amber → orange → red where the photo is evidence *for* the explained class, light
  blue → indigo where it is evidence *against* it (the SHAP convention), at most 66% opaque so the
  features under the hot spot stay visible; a white outline around the focus region (the smallest
  region holding half of the evidence); or the photo dimmed outside that region. Grad-CAM maps are
  smoothed with half-pixel-centre bilinear upsampling; occlusion / LIME region maps stay blocky.
- `explanation_summary(explanation)` returns the API's plain-English `options["summary"]` (and builds
  one from the heatmap for older records; regression records speak of "the prediction" with three
  significant figures). `evidence_images(image, explanation)` returns the photo, the photo with its
  focus hidden and the photo with only the focus kept — the pictures behind `options["evidence"]`,
  hidden the way the API hid them: blurred when `evidence["fill"] == "blur"`, grayed otherwise.
- `gaussian_blur(image, sigma)`: the API's blur for hiding regions (radius `ceil(3σ)`, mirrored
  edges, separable, float64), NumPy only and identical to the server's to 1e-12.
- `numbered_regions(explanation, for_count=5, against_count=3)`: the regions to call out, numbered
  1..N across two groups — the strongest *for* the class, then the strongest *against* it (Grad-CAM:
  its strongest regions only) — each with its name, attribution and `anchor` (a point inside the
  region for a number badge).
- `focus_mask`, `region_mask(explanation, name_or_id, h, w)` and `focus_threshold` expose the masks;
  `upsample_heatmap`, `heat_color` and `heat_alpha` are the building blocks. All of it is NumPy +
  Pillow and byte-identical to the dashboard (`tests/fixtures/heatmap_render.json` pins the shared
  cases; blurred evidence thumbnails may differ from a JavaScript renderer by one byte), and
  `import kaimei.images` loads neither Pillow nor matplotlib.
- `plot_explanation(image, explanation, *, modes=("heatmap", "spotlight"), figure=None, title=None,
  for_regions=5, against_regions=3)` draws the dashboard's view with matplotlib (optional): the
  photo with the model's answer, the heatmap with numbered badges on the strongest regions, and the
  spotlight; below, the evidence check (full photo, focus blurred or hidden, only the focus) with
  the model's probability for each; on the right, spanning both rows, a bar chart of the same
  numbered regions ("For cat" in orange, "Against cat" in blue) with a plain sentence saying what a
  bar measures. The summary is the title, wrapped to the actual figure or subfigure width.
  `BAR_POSITIVE` / `BAR_NEGATIVE` (`#f97316` / `#5b5bf5`) and `SUPPORT_GRADIENT` /
  `AGAINST_GRADIENT` are the matching chart colours.
- `examples/photos/`: six Oxford-IIIT Pet test photos (three cats, three dogs) and a cat-and-dog
  composite, with credits. `examples/cats_dogs_gradcam.py upload` attaches them to the image feature
  as clickable dashboard examples, the new `examples` subcommand attaches them to a model that is
  already registered, and `explain` prints the summary, the evidence check and the numbered regions
  and saves the full figure.
- `examples/cats_dogs_explain.ipynb`: Grad-CAM on a cat and a dog, "why cat?" and "why dog?" with
  LIME on the cat-and-dog composite, Grad-CAM, occlusion and LIME side by side on the beagle (and
  why perturbation methods can disagree with Grad-CAM), and a Grad-CAM + LIME gallery of all six
  photos, with the narrative written from the executed outputs. Figures are inlined as JPEG so the
  notebook stays small.

### Changed

- `overlay_heatmap(image, heatmap)` keeps its signature but draws the new look (graded glow, focus
  outline) and gains `method`, `mode`, `threshold` and `outline`; `opacity` is accepted and ignored.
  `method=None` (the default) infers the upsampling from the values: a map with any negative value
  is a region map (nearest), anything else Grad-CAM (bilinear). Bytes are rounded half-up
  (`floor(x · 255 + 0.5)`) so every renderer agrees.
- The `dev` extra installs matplotlib, so CI runs the `plot_explanation` tests instead of skipping
  them. matplotlib stays optional for users.

## 0.4.0 — 2026-10-24

Convolutional image models upload exactly, from PyTorch and Keras, and photos load and explain
from Python.

### Added

- PyTorch CNNs are traced with `torch.fx` (forward hooks remain the fallback), so residual adds,
  `torch.cat`, squeeze-excitation gates, `torch.flatten` and functional activations are kept.
  `Conv2d` with any stride / padding / dilation / groups, `BatchNorm2d`, max / average /
  adaptive pooling, `ZeroPad2d`, `ReLU6`, `LeakyReLU`, `ELU`, `Hardswish`, `Hardsigmoid`.
  ResNet, MobileNet v2 / v3, EfficientNet, DenseNet, VGG and SqueezeNet replay to within 1e-7 of
  PyTorch. Pass `input_shape=(channels, height, width)`.
- Keras functional graphs (and nested base models, as in transfer learning) keep their branches:
  `DepthwiseConv2D`, `SeparableConv2D`, `BatchNormalization`, average and global pooling, `Add`,
  `Multiply`, `Concatenate`, `ZeroPadding2D`, `Rescaling`, `Normalization`, strided and dilated
  `Conv2D` with `"same"` padding. MobileNetV2, ResNet50 and EfficientNetB0 replay exactly.
- `preprocess={"mean": [...], "std": [...]}` bakes input normalization into the graph;
  `value_range=[0, 255]` declares the pixel units an image model reads.
- Classifiers that end in logits get a softmax / sigmoid appended so Kaimei scores probabilities.
- `upload(X=...)` accepts image batches, `(n, h, w, c)` or `(n, c, h, w)`, and keeps the reference
  set under the API's value budget.
- `client.projects.list(include="summary")` / `retrieve(..., include="summary")` return each
  project's roll-up (models, 30-day traffic, alerts), and `client.projects.overview(project)` returns
  the project dashboard: per-model traffic and drift, open alerts, the feature × model attribution
  matrix, and recent analyses and explanations.
- `kaimei.images.load_image(source, size=None, *, model=None)` reads a photo (a path, bytes, a PIL
  image or an array) into the nested list an image feature expects: the shorter side scaled to fit
  and the middle cropped, exactly as torchvision's `Resize` + `CenterCrop`, with the EXIF rotation
  applied and pixels in the feature's `value_range`. `model=` takes the size from a model record.
- `kaimei.images.overlay_heatmap(image, heatmap)` paints an explanation's `options["heatmap"]` over
  the image, as the dashboard draws it.
- `pip install "kaimei[images]"` installs NumPy and Pillow for these helpers.
- `examples/cats_dogs_gradcam.py`: train a MobileNetV2 cat/dog classifier on the Oxford-IIIT Pet
  photos, upload it, and run Grad-CAM on new photos.

### Changed

- An operation that cannot be replayed and changes the tensor shape raises
  `StructureError(code="unsupported_layers")` naming it, instead of an approximate graph that
  could not compile.
- Request bodies may contain NumPy arrays and scalars; they are sent as plain JSON lists and numbers.
- With the default `externalize_above`, a spec that would still be 2 MB of JSON or more (a CNN's
  many mid-sized kernels: MobileNetV2 was 16 MB) also lifts tensors above 4,096 values to binary,
  so it uploads as 1.4 MB. A threshold you pass is used as given.

## 0.3.0 — 2026-10-23

Models with millions of parameters and thousands of layers upload and render.

### Added

- Tensors with more than `externalize_above` values (default 65,536) leave the JSON spec and
  upload as zlib-compressed float32 blobs (`PUT /structure/tensors/{key}`), referenced from the
  spec as `{"$tensor": key}`. `client.models.upload()` registers the model from its shapes,
  streams the tensors, then sends the real spec; `progress=` reports each stage.
- `weights="summary"` uploads shapes and statistics only. The dashboard draws the architecture
  while the weights stay in your environment.
- `extract_structure()` returns an `Extraction` (spec plus tensors); `estimator_to_spec()` is
  unchanged and keeps everything inline.
- PyTorch and Keras layers carry their module `path`, so the API groups them into blocks and
  detects repeated blocks (`×N`).
- `client.models.tensors()`, `delete_tensor()`, `structure_parameter_values()` (binary window),
  `structure(root=, depth=, detail=)` and `trace_structure(detail="summary", nodes=)`.

## 0.2.2 — 2026-10-13

### Fixed

- mypy type-checks as Python 3.10 and resolves feature names to `list[str]` before
  reading a wrapped estimator.

## 0.2.1 — 2026-10-13

### Added

- `estimator_to_spec` unwraps wrappers around a fitted sklearn / Keras / Torch estimator
  (`model_`, `estimator_`, `best_estimator_`, `module`, and the same family) without
  special-casing vendor packages. Framework labels follow the core-library MRO.

## 0.2.0 — 2026-10-12

First release that can upload a fitted estimator from a notebook.

### Added

- `client.models.upload()` extracts sklearn / Keras / PyTorch structure and registers it in one call.
- `estimator_to_spec`, reference-data helpers, and a supported-estimator matrix in the README.
- Structure-backed inference on the API: executable uploads can be explained, predicted, and analysed without a remote URL.
- `client.health()`, `Page.auto_paging_iter()`, and `*.iterate()` helpers.
- Retries on 429 (and GET 502/503/504), honouring `Retry-After`.
- `auth.signup(..., turnstile_token=)` plus `verify_email` / `resend_verification`.
- Examples: `examples/upload_sklearn.py` and `examples/upload_sklearn.ipynb`.
- `PermissionDeniedError` (keeps `PermissionError` as an alias).

### Changed

- Version bump from 0.1.0. Default HTTP timeout is 120 seconds.
- `upload()` no longer PUTs `/structure` after `POST /v1/models` already stored it.
- Reference rows are capped at 5,000 with a warning. Large JSON specs warn at 2 MB and 16 MB.
- Non-executable uploads (importance summaries, approximate nets) warn that explain/predict need a remote URL.

### Packaging

- Extra `kaimei[models]` installs NumPy.
- CI runs Ruff, mypy, and Python 3.9–3.13.
