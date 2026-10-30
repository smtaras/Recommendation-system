# Retailrocket Recommendation System

Pet project for exploring two recommendation approaches on the Retailrocket e-commerce dataset:

- a related-item recommender based on products added to cart in the same session;
- a personalized ALS recommender based on a visitor's past interactions.

The notebooks contain the data exploration, modelling, and evaluation. The local API serves the personalized ALS model and falls back to popular items for visitors not present in the training data.

## Project structure

```text
.
├── 01_related_item_recommendations.ipynb
├── 02_personalized_recommendations.ipynb
├── recommender.py
├── app.py
├── requirements.txt
├── data/                    # local Kaggle CSVs; not committed
└── artifacts/               # trained model, sparse matrix, and metadata
```

## Dataset

The notebooks use the [Retailrocket eCommerce dataset on Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset). The local `data/` directory should contain:

- `events.csv`
- `item_properties_part1.csv` and `item_properties_part2.csv` for the category-based steps in Notebook 01;
- `category_tree.csv` is included in the Kaggle download, but the current notebooks do not read it.

Both notebooks read `events.csv` from `data/`; Notebook 01 also reads the two item-property files. The first notebook downloads the Kaggle dataset with `kagglehub` and prints the cache path, but does not automatically copy those files into `data/`. Place the required CSVs there before running the notebooks. The API does not need the raw CSVs once the model artifacts exist.

The dataset contains event and item identifiers, not a human-readable product catalogue. `item_properties` includes a numeric `categoryid`; `category_tree.csv` describes parent-child relationships between category IDs, but does not provide readable category names. The API therefore returns item IDs. A storefront would need to join those IDs with its own catalogue to display product names, images, or descriptions.

## Notebook 01 — Related items

[`01_related_item_recommendations.ipynb`](01_related_item_recommendations.ipynb) builds a complementary/similar-item recommender:

- Loads `events.csv` and keeps `addtocart` events for co-occurrence analysis.
- Creates visitor-specific cart sessions. Two consecutive cart events belong to the same session when their timestamp gap is at most 1,000,000 ms (16 minutes 40 seconds).
- Creates unordered item pairs within each session, counts how often each pair co-occurs, and retains pairs seen in at least three sessions.
- Filters sessions with more than 20 unique items. In the notebook's data check, these were about 0.05% of sessions but generated about 27.29% of all possible pairs; filtering reduced the pair set from 52,917 to 38,477 and frequent pairs from 548 to 464.
- Extracts `categoryid` rows from both item-property files and uses the latest recorded category per item. For items with category data, same-category products are ranked by add-to-cart popularity.
- `get_product_recommendations(item_id, complementary_n=3, top_n=5)` prioritizes up to three co-occurring complementary items, then fills the remaining slots with popular items from the same category.

This is a heuristic recommender, not a trained ranking model. Co-occurrence is used as a proxy for complementarity; same-category membership is a proxy for similarity. Its output is not personalized to a visitor. Category IDs have no readable labels in this dataset, and the function may return fewer than five items when there are not enough candidates.

## Notebook 02 — Personalized recommendations

[`02_personalized_recommendations.ipynb`](02_personalized_recommendations.ipynb) trains and evaluates an implicit-feedback ALS model:

- Explores `view`, `addtocart`, and `transaction` event volumes and the event time range (2015-05-03 to 2015-09-18 UTC).
- Uses a chronological 80/20 split. The outer training period ends at 2015-08-18 04:23:31 UTC; later events are held out as the outer test period.
- Gives event types strengths of `view=1`, `addtocart=3`, and `transaction=5`. For each visitor-item pair, the strongest event is kept (`max`), rather than summing repeated events.
- Builds a sparse visitor-by-item matrix and fits implicit Alternating Least Squares (ALS). The selected serving configuration is `factors=128`, `regularization=0.5`, `alpha=40`, `iterations=10`, `random_state=42`.
- Evaluates top-10 recommendations for warm visitors and future add-to-cart/transaction targets. Previously seen visitor-item pairs and items absent from the training catalogue are excluded from the target set.
- Compares ALS with a popularity baseline and tests two chronological validation windows while selecting `alpha`. The 30-day recency-decay experiment and raising the transaction event weight to 10 did not provide a consistent benefit, so the serving model retains weights `1/3/5` without recency decay.

The main metric is macro Recall@10: for each evaluated visitor, it measures the fraction of that visitor's unique future target items found among the ten recommendations, then averages over visitors. It is not the probability that a recommendation will be clicked or purchased.

### Recorded evaluation snapshot

The saved notebook currently reports the following outer-test results for the selected model:

| Future event | ALS Recall@10 | Popularity Recall@10 |
| --- | ---: | ---: |
| Add to cart | 0.0214 | 0.0088 |
| Transaction | 0.0134 | 0.0171 |

ALS beats the popularity baseline for add-to-cart targets in this run, but not for transactions. On the outer test period, only about 8% of distinct visitors had appeared in training; the notebook's offline ALS metric is evaluated on warm visitors and known items. New visitors are handled by the API's popularity fallback, but the displayed offline metrics do not evaluate that fallback over every new visitor.

Validation scores varied between time windows, particularly for transactions. The test period was also inspected during iterative development, so treat these numbers as a project evaluation snapshot rather than a pristine, untouched benchmark. Do not tune model parameters against the outer test results.

The two validation windows used for the `alpha` comparison were:

| Validation period | `alpha=40` cart | `alpha=40` transaction | `alpha=80` cart | `alpha=80` transaction |
| --- | ---: | ---: | ---: | ---: |
| 2015-07-26 to 2015-08-18 | 0.0532 | 0.0735 | 0.0571 | 0.0692 |
| 2015-07-07 to 2015-07-26 | 0.0597 | 0.0386 | 0.0577 | 0.0386 |

These windows have different training histories and target populations, so compare each configuration within a window rather than interpreting the two rows as directly equivalent test populations. `alpha=40` was retained as the balanced choice across the two windows.

## Recommendation service and API

[`recommender.py`](recommender.py) defines `Recommender`, which loads the saved ALS model, sparse user-item matrix, and metadata once when instantiated.

For a request, `Recommender.recommend_items(visitor_id, n=10, exclude_item_ids=None)`:

- uses ALS for a visitor ID found in the training map;
- filters items already present in that visitor's training history;
- applies any explicit exclusions (for example, items in the current session);
- fills any remaining slots with globally popular unseen items;
- uses popularity-only recommendations when the visitor is unknown.

[`app.py`](app.py) exposes this functionality through FastAPI:

- `GET /health` returns `{"status": "ok"}`;
- `POST /recommendations` accepts `visitor_id`, optional `n` (1–50), and optional `exclude_item_ids`;
- the response includes `visitor_id`, a `strategy` (`als` or `popular_fallback`), and a list of recommended item IDs.

Example request body:

```json
{
  "visitor_id": 123456,
  "n": 10,
  "exclude_item_ids": []
}
```

FastAPI validates the request body and serves interactive API documentation at `/docs`.

The serving flow is:

```text
Kaggle event data
    → chronological split and event weighting
    → sparse visitor-item matrix
    → ALS model and ID mappings
    → saved artifacts
    → Recommender class
    → FastAPI JSON response
```

## Setup and run

Use the same Python environment for the notebook and API. The project has been exercised in a Python 3.12 environment.

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

To rebuild the model artifacts, run the data preparation and final training/export cells in Notebook 02 in order. The exploratory ALS experiments take additional time. The API requires the following files under `artifacts/`:

- `als_model.npz` — trained ALS factors (currently roughly 684 MB);
- `user_item_matrix.npz` — sparse training matrix used for seen-item filtering;
- `metadata.pkl` — visitor/item index mappings, popularity fallback list, and model configuration.

Start the local API from the project root:

```powershell
python -m uvicorn app:app --reload
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), try `POST /recommendations`, and inspect the response. Stop the development server with `Ctrl+C`.

## Dependencies

See [`requirements.txt`](requirements.txt). `kagglehub` is only needed for the notebook's Kaggle download step; FastAPI and Uvicorn are needed to run the API; `ipykernel` is included for notebook use in editors such as VS Code.

## Current limitations

- Recommendations are item IDs only; product names, images, prices, and readable category names are not included in the dataset.
- The personalized ALS model cannot personalize for a visitor absent from training; those requests use popularity fallback.
- The related-item recommender and personalized ALS service are separate systems. The current API serves the personalized ALS recommender only.
- The project is a local development MVP. It does not yet include authentication, deployment configuration, a product-catalog integration, automated API tests, or monitoring.