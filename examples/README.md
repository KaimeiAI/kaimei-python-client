# Examples: models that are not tables

Kaimei client 0.12.0 uploads models that read text or pictures, not only tables. Each example is one file read from top to bottom, as a script and as a notebook with the same code.

| Example | What it shows |
| --- | --- |
| [`text_sentiment.py`](text_sentiment.py) / [`.ipynb`](text_sentiment.ipynb) | A scikit-learn `Pipeline` (TF-IDF, then logistic regression) trained on 60 short reviews written in the file. It is uploaded as it is; Kaimei reads it exactly and the model then takes text. One prediction, and one explanation whose parts are the review's words. |
| [`image_segmentation.py`](image_segmentation.py) / [`.ipynb`](image_segmentation.ipynb) | A tiny PyTorch convolutional segmenter trained for a few seconds on synthetic 32 x 32 pictures (a bright disc on noise). It is uploaded with `task="segmentation"`; one prediction, whose mask is compared with the model's own (sigmoid at 0.5 or more) and printed as a text grid. |

Nothing is downloaded, and both fit the free plan (one model run by an agent, graphs up to 25 million numbers; these are a few thousand).

## Run them

```bash
pip install kaimei                    # 0.12.0 or later, against Kaimei API 1.3.0 or later
pip install -r examples/requirements.txt
export KAIMEI_API_KEY=km_live_...     # a key from the dashboard
export KAIMEI_BASE_URL=http://localhost:8000    # optional: only for a self-hosted API
python examples/text_sentiment.py
python examples/image_segmentation.py
```

On Windows set the variables with `set KAIMEI_API_KEY=...`. For the notebooks, `pip install jupyter` and open the `.ipynb` files; they are saved with the outputs of one run.

Each run uploads one model to your workspace. The free plan holds 10 models, so delete the ones you do not want (`client.models.delete(model.id)`).

## What you will see

`python examples/text_sentiment.py`:

```
The pipeline was trained on 60 reviews and scores 100% on them.
examples/text_sentiment.py:101: UserWarning: X was not uploaded as reference data: this model reads text, a series or a tensor, and reference rows are stored for tables and images.
  model = client.models.upload(
The model is registered as mdl_....
Kaimei's reading is exact (check: verified); it runs the model.
Kaimei predicts positive (68.1%) for 'a clever and moving film, but the ending was weak'.
The pipeline itself says [0.3188, 0.6812] (negative, positive).
The words that mattered most, with their attribution toward the predicted class:
  clever     +0.249
  weak       -0.238
  moving     +0.207
  film       -0.029
  and        -0.024
```

The warning is the client saying the reviews were used to check the reading but not stored as reference data. The attributions come from sampled permutations, so they move by a hundredth or two from run to run (the notebook's saved run has `clever +0.264`).

`python examples/image_segmentation.py`:

```
The network was trained for 150 steps; its last loss was 0.007.
The model is registered as mdl_....
Kaimei's reading is exact (check: verified); it runs the model.
Kaimei's mask agrees with the model's own on 1024 of 1024 pixels.
The model marks 57 pixels as disc; the true disc has 57.
Kaimei's mask (# disc, . background), every second row:
................................
................................
................................
................................
................................
................................
.....................#######....
....................#########...
.....................########...
.......................###......
................................
................................
................................
................................
................................
................................
```

"Exact" and "verified" are the structure's `fidelity` and `verification`: the API ran the uploaded structure on the rows that came with the model and got the library's own answers. The picture is random, so yours will differ.

## Running a model with your own agent instead

Not executed here. When the weights must stay in your network, an agent beside the model answers Kaimei's requests (see "Models that stay with you: the agent" in the main README).

```python
from kaimei.agent import serve

model = client.models.upload(net, name="Disc segmenter", task="segmentation", sample=sample, weights="summary",
                             inference={"type": "agent"}, save_onnx="segmenter.onnx")
serve(client, model.id, onnx="segmenter.onnx")   # blocks; Ctrl+C stops it
```
